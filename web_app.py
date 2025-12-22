import streamlit as st
from supabase import create_client, Client
import re
import requests
import time
import random
import math
from datetime import datetime
from collections import Counter
import yt_dlp
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="Fact-Check Center v48.0 (Hybrid)", layout="wide", page_icon="⚖️")

# 🌟 Secrets에서 키 가져오기
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except:
    st.error("❌ 필수 키가 설정되지 않았습니다. Streamlit Secrets를 확인해주세요.")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [관리자 인증] ---
if "is_admin" not in st.session_state: st.session_state["is_admin"] = False
with st.sidebar:
    st.header("🛡️ 관리자 메뉴")
    with st.form("login_form"):
        password_input = st.text_input("관리자 비밀번호", type="password")
        if st.form_submit_button("로그인"):
            if password_input == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True; st.rerun()
            else: st.session_state["is_admin"] = False; st.error("불일치")
    if st.session_state["is_admin"]:
        st.success("✅ 관리자 인증됨"); 
        if st.button("로그아웃"): st.session_state["is_admin"] = False; st.rerun()

# --- [상수 설정] ---
WEIGHT_NEWS_DEFAULT = 45       
WEIGHT_VECTOR = 35     
WEIGHT_CONTENT = 15    
WEIGHT_SENTIMENT_DEFAULT = 10  
PENALTY_ABUSE = 20     
PENALTY_MISMATCH = 30
PENALTY_NO_FACT = 25
PENALTY_SILENT_ECHO = 40  

# 핵심 상태어 (정치/사회적 파급력이 큰 단어)
VITAL_KEYWORDS = [
    '위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '쓰러져', 
    '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', 
    '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '내란'
]

# VIP 인물 사전
VIP_ENTITIES = [
    '윤석열', '대통령', '이재명', '한동훈', '김건희', '문재인', '박근혜', '이명박',
    '트럼프', '바이든', '푸틴', '젤렌스키', '시진핑', '정은', 
    '이준석', '조국', '추미애', '홍준표', '유승민', '안철수',
    '손흥민', '이강인', '김민재', '류현진', '재용', '정의선', '최태원'
]

# 공식 언론사 리스트 (뉴스감별사 등 유사 언론 제외)
OFFICIAL_CHANNELS = [
    'MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS',
    '채널A', 'TV조선', '연합뉴스', 'YONHAP',
    '한겨레', '경향', '조선일보', '중앙일보', '동아일보', '한국일보', '국민일보', 
    '서울신문', '세계일보', '문화일보', '매일경제', '한국경제', '서울경제',
    'CHOSUN', 'JOONGANG', 'DONGA', 'HANKYOREH', 'KYUNGHYANG'
]

STATIC_TRUTH_CORPUS = ["박나래 위장전입 의혹 무혐의", "임영웅 콘서트 암표 대응", "정희원 교수 저속노화", "대전 충남 행정 통합", "선거 출마 선언", "강훈식 의원 출마설"]
STATIC_FAKE_CORPUS = ["충격 폭로 경악", "긴급 속보 소름", "이재명 한동훈 충격 발언", "결국 구속 영장 발부", "방송 불가 영상 유출", "꿈속 계시 예언", "사형 선고 집행", "건강 악화 위독설"]

class VectorEngine:
    def __init__(self):
        self.vocab = set(); self.truth_vectors = []; self.fake_vectors = []
    def tokenize(self, text): return re.findall(r'[가-힣]{2,}', text)
    def build_vocabulary(self, corpus):
        for text in corpus: self.vocab.update(self.tokenize(text))
        self.vocab = sorted(list(self.vocab))
    def text_to_vector(self, text):
        tokens = self.tokenize(text); token_counts = Counter(tokens); vector = []
        for word in self.vocab: vector.append(token_counts[word])
        return vector
    def cosine_similarity(self, vec1, vec2):
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1)); mag2 = math.sqrt(sum(b * b for b in vec2))
        return dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0.0
    def train(self, truth, fake):
        self.build_vocabulary(truth + fake)
        self.truth_vectors = [self.text_to_vector(t) for t in truth]
        self.fake_vectors = [self.text_to_vector(t) for t in fake]
    def analyze_position(self, query):
        q_vec = self.text_to_vector(query)
        max_t = max([self.cosine_similarity(q_vec, v) for v in self.truth_vectors] or [0])
        max_f = max([self.cosine_similarity(q_vec, v) for v in self.fake_vectors] or [0])
        return max_t, max_f

vector_engine = VectorEngine()

def save_analysis(channel, title, prob, url, keywords):
    try: supabase.table("analysis_history").insert({
        "channel_name": channel, "video_title": title, "fake_prob": prob, 
        "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
        "video_url": url, "keywords": keywords}).execute()
    except: pass

def train_dynamic_vector_engine():
    try:
        dt = [r['video_title'] for r in supabase.table("analysis_history").select("video_title").lt("fake_prob", 30).execute().data]
        df = [r['video_title'] for r in supabase.table("analysis_history").select("video_title").gt("fake_prob", 70).execute().data]
    except: dt, df = [], []
    vector_engine.train(STATIC_TRUTH_CORPUS + dt, STATIC_FAKE_CORPUS + df)
    return len(STATIC_TRUTH_CORPUS + dt) + len(STATIC_FAKE_CORPUS + df)

# --- [Advanced Logic Functions] ---

# 🌟 [개선] 노이즈 리스트 대폭 강화 (일반적인 단어 제외)
def get_noise_words():
    return ['충격', '경악', '실체', '난리', '공개', '반응', '명단', '동영상', '사진', '집안', '속보', 
            '단독', '결국', 'MBC', '뉴스', '이미지', '너무', '다른', '알고보니', 'ㄷㄷ', '진짜', 
            '정말', '영상', '사람', '생각', '오늘밤', '오늘', '내일', '지금', '못넘긴다', '넘긴다', 
            '이유', '왜', '안', '대부분', '모르지만', '있는', '없는', '하는', '몸을', '몸', '건강']

def extract_nouns(text):
    noise = get_noise_words()
    nouns = re.findall(r'[가-힣A-Za-z0-9]{2,}', text)
    return [n for n in nouns if n not in noise]

# 🌟 [v48.0 신규] 하이브리드 키워드 추출 (제목 + 자막 융합)
def generate_hybrid_query(title, hashtags, transcript):
    # 1. 소스 준비
    title_text = title + " " + " ".join([h.replace("#", "") for h in hashtags])
    transcript_text = transcript if transcript else ""
    
    # 2. 제목에서 키워드 추출 (가중치 높음)
    title_nouns = extract_nouns(title_text)
    
    # 3. 자막에서 빈출 키워드 추출 (가중치 낮음, 보완용)
    transcript_nouns = extract_nouns(transcript_text)
    transcript_counter = Counter(transcript_nouns)
    # 상위 3개만 추출 (너무 많으면 검색 꼬임)
    top_transcript_nouns = [word for word, count in transcript_counter.most_common(3)]
    
    # 4. VIP/Vital 체크 (최우선 순위)
    vip_found = [vip for vip in VIP_ENTITIES if vip in title_text]
    vital_found = [vital for vital in VITAL_KEYWORDS if vital in title_text]
    
    # 5. 쿼리 조합 로직
    final_query = []
    
    if vip_found:
        # VIP가 있으면: VIP + Vital + (제목 명사 중 VIP 아닌 것)
        final_query.extend(vip_found)
        if vital_found: final_query.extend(vital_found)
        # VIP 관련 문맥 추가 (조사 기반 Chunking 로직 간소화 적용)
        for t_noun in title_nouns:
            if t_noun not in final_query and t_noun not in VIP_ENTITIES:
                final_query.append(t_noun)
                break # 하나만 추가
    else:
        # VIP가 없으면: 제목 명사 + 자막 빈출 명사 결합
        final_query.extend(title_nouns[:2]) # 제목 앞 2개
        
        # 자막에서 보완 (제목에 없는 내용이면 추가)
        for tr_noun in top_transcript_nouns:
            if tr_noun not in final_query:
                final_query.append(tr_noun)
                if len(final_query) >= 3: break # 최대 3단어 조합
                
    return " ".join(final_query)

def summarize_transcript(text, max_sentences=3):
    if not text or len(text) < 50: return "⚠️ 요약할 자막 내용이 충분하지 않습니다."
    sentences = re.split(r'(?<=[.?!])\s+', text)
    if len(sentences) <= max_sentences: return text
    nouns = re.findall(r'[가-힣]{2,}', text); word_freq = Counter(nouns); ranked_sentences = []
    for i, sent in enumerate(sentences):
        sent_nouns = re.findall(r'[가-힣]{2,}', sent)
        if not sent_nouns: continue
        score = sum(word_freq[w] for w in sent_nouns)
        if 10 < len(sent) < 150: ranked_sentences.append((i, sent, score / len(sent_nouns)))
    top_sentences = sorted(ranked_sentences, key=lambda x: x[2], reverse=True)[:max_sentences]
    top_sentences.sort(key=lambda x: x[0])
    return f"📌 **핵심 요약**: {' '.join([s[1] for s in top_sentences])}"

def clean_html(raw_html): return BeautifulSoup(raw_html, "html.parser").get_text()

def detect_ai_content(info):
    is_ai, reasons = False, []
    ai_keywords = ['ai', 'artificial intelligence', 'chatgpt', 'midjourney', 'sora', 'deepfake', 'synthetic', '인공지능', '딥페이크', '가상인간', '버추얼', 'gpt']
    text_to_check = (info.get('title', '') + " " + info.get('description', '') + " " + " ".join(info.get('tags', []))).lower()
    for kw in ai_keywords:
        if re.search(r'\b{}\b'.format(re.escape(kw)), text_to_check): is_ai = True; reasons.append(f"키워드 감지: {kw}"); break
    return is_ai, ", ".join(reasons)

def check_is_official(channel_name):
    norm_name = channel_name.upper().replace(" ", "")
    for official in OFFICIAL_CHANNELS:
        if official in norm_name: return True
    return False

def count_sensational_words(text):
    triggers = ['충격', '경악', '실체', '폭로', '난리', '속보', '긴급', '소름', 'ㄷㄷ', '진짜', '결국', '계시', '예언', '위독', '사망', '중태']
    count = 0
    for w in triggers: count += text.count(w)
    return count

def check_tag_abuse(title, hashtags, channel_name):
    if check_is_official(channel_name): return 0, "공식 채널 면제"
    if not hashtags: return 0, "해시태그 없음"
    title_nouns = extract_nouns(title); tag_nouns = set()
    for t in hashtags: tag_nouns.add(t.replace("#", "").split(":")[-1].strip())
    if len(tag_nouns) < 2: return 0, "양호"
    if not set(title_nouns).intersection(tag_nouns): return PENALTY_ABUSE, "🚨 심각 (불일치)"
    return 0, "양호"

def fetch_real_transcript(info_dict):
    sub_url = None
    if 'subtitles' in info_dict and 'ko' in info_dict['subtitles']:
        for fmt in info_dict['subtitles']['ko']:
            if fmt['ext'] == 'vtt': sub_url = fmt['url']; break
    if not sub_url and 'automatic_captions' in info_dict and 'ko' in info_dict['automatic_captions']:
        for fmt in info_dict['automatic_captions']['ko']:
            if fmt['ext'] == 'vtt': sub_url = fmt['url']; break
    if not sub_url: return None, "자막 없음 (설명란 대체)"
    try:
        response = requests.get(sub_url)
        if response.status_code == 200:
            lines = response.text.splitlines(); clean_lines = []; seen = set()
            for line in lines:
                line = line.strip()
                if '-->' in line or line == 'WEBVTT' or not line: continue
                line = re.sub(r'<[^>]+>', '', line)
                if line and line not in seen: clean_lines.append(line); seen.add(line)
            return " ".join(clean_lines), "✅ 실제 자막 수집 성공"
    except: pass
    return None, "자막 다운로드 실패"

def fetch_comments_via_api(video_id):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {'part': 'snippet', 'videoId': video_id, 'key': YOUTUBE_API_KEY, 'maxResults': 100, 'order': 'relevance'}
    try:
        res = requests.get(url, params=params)
        if res.status_code == 200:
            items = res.json().get('items', [])
            top_comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in items]
            return top_comments[:50], f"✅ API 수집 성공 (Top {len(top_comments[:50])} by Likes)"
        elif res.status_code == 403: return [], "⚠️ API 권한 오류"
        elif res.status_code == 404: return [], "⚠️ 댓글 사용 중지됨"
        else: return [], f"⚠️ API 오류 ({res.status_code})"
    except: return [], f"❌ API 통신 실패"

def calculate_dual_match(news_item, query_nouns, transcript):
    if not news_item or not transcript: return 0, 0, 0
    news_title = news_item.get('title', ''); title_nouns = extract_nouns(news_title)
    intersection = len(set(query_nouns).intersection(set(title_nouns)))
    title_score = 1.0 if intersection >= 2 else (0.5 if intersection == 1 else 0)
    news_desc = news_item.get('desc', ''); desc_nouns = extract_nouns(news_desc)
    if not desc_nouns: content_score = 0
    else:
        match_count = 0
        for noun in desc_nouns:
            if noun in transcript: match_count += 1
        content_ratio = match_count / len(desc_nouns)
        content_score = 1.0 if content_ratio >= 0.3 else (0.5 if content_ratio >= 0.15 else 0)
    total_score = (title_score * 0.3) + (content_score * 0.7)
    return int(total_score * 100), int(title_score * 100), int(content_score * 100)

def analyze_comment_relevance(comments, context_text):
    if not comments: return [], 0, "분석 불가"
    all_comments_text = " ".join(comments); comment_nouns = extract_nouns(all_comments_text)
    if not comment_nouns: return [], 0, "유효 키워드 없음"
    top_keywords = Counter(comment_nouns).most_common(5)
    context_nouns = extract_nouns(context_text); match_count = 0; context_set = set(context_nouns)
    for word, cnt in top_keywords:
        if word in context_set: match_count += 1
    relevance_score = int((match_count / len(top_keywords)) * 100)
    msg = "✅ 주제 집중" if relevance_score >= 60 else "⚠️ 일부 관련" if relevance_score >= 20 else "❌ 무관/잡담"
    return [f"{w}({c})" for w, c in top_keywords], relevance_score, msg

def check_red_flags(comments):
    keywords = ['가짜뉴스', '가짜 뉴스', '주작', '사기', '거짓말', '허위', '구라', '합성', '선동', '소설']
    count = 0; detected = []
    for c in comments:
        for k in keywords:
            if k in c: count += 1; detected.append(k)
    return count, list(set(detected))

# --- [Main] ---
def run_forensic_main(url):
    total_intelligence = train_dynamic_vector_engine()
    witty_loading_sequence(total_intelligence)
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if video_id: video_id = video_id.group(1)
    
    ydl_opts = {'quiet': True, 'skip_download': True, 'writesubtitles': True, 'subtitleslangs': ['ko'], 'extractor_args': {'youtube': {'skip': ['dash', 'hls']}}}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', ''); uploader = info.get('uploader', '')
            tags = info.get('tags', []); desc = info.get('description', '')
            
            transcript_text, transcript_status = fetch_real_transcript(info)
            analysis_text = transcript_text if transcript_text else desc
            
            # 🌟 [v48.0] 하이브리드 쿼리 생성
            refined_query = generate_hybrid_query(title, tags, transcript_text)
            
            is_official = check_is_official(uploader)
            is_ai_content, ai_reason = detect_ai_content(info)
            abuse_score, abuse_status = check_tag_abuse(title, tags, uploader)
            summary_text = summarize_transcript(analysis_text)
            agitation_count = count_sensational_words(analysis_text + title)
            agitation_level = "높음 (위험)" if agitation_count > 3 else "보통"
            
            # 벡터 분석
            t_sim, f_sim = vector_engine.analyze_position(refined_query + " " + title)
            t_impact = int(t_sim * 35) * -1; f_impact = int(f_sim * 35)
            
            # 뉴스 검색
            news_ev = []; max_dual_score = 0; news_cnt = 0
            try:
                rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(refined_query)}&hl=ko&gl=KR"
                root = ET.fromstring(requests.get(rss_url, timeout=5).content)
                items = root.findall('.//item'); news_cnt = len(items)
                for item in items[:3]:
                    nt = item.find('title').text; nd = clean_html(item.find('description').text)
                    total, t_sc, c_sc = calculate_dual_match({'title': nt, 'desc': nd}, extract_nouns(refined_query), analysis_text)
                    if total > max_dual_score: max_dual_score = total
                    news_ev.append({"뉴스 제목": nt, "일치도": f"{total}%"})
            except: pass
            
            # 댓글 분석
            comments, c_status = fetch_comments_via_api(video_id)
            top_k, rel_score, rel_msg = analyze_comment_relevance(comments, title + " " + analysis_text)
            red_cnt, red_words = check_red_flags(comments)
            is_controversial = red_cnt > 0
            
            # 점수 산정
            news_score = 0; silent_penalty = 0; mismatch_penalty = 0
            is_effective_silence = (news_cnt == 0) or (news_cnt > 0 and max_dual_score < 20)
            
            if is_effective_silence:
                if agitation_count >= 3: silent_penalty = PENALTY_SILENT_ECHO; t_impact *= 2; f_impact *= 2
                else: mismatch_penalty = 10 
            elif is_controversial:
                if max_dual_score < 60: news_score = PENALTY_NO_FACT 
                else: news_score = int((max_dual_score/100)**2 * 65) * -1
            else:
                news_score = int((max_dual_score/100)**2 * 45) * -1
            
            if is_official: news_score = -50; mismatch_penalty = 0; silent_penalty = 0
            
            total_score = 50 + t_impact + f_impact + news_score + silent_penalty + mismatch_penalty + abuse_score
            final_prob = max(5, min(99, total_score))
            
            save_analysis(uploader, title, final_prob, url, refined_query)
            
            # --- UI Output ---
            st.subheader("🕵️ 핵심 분석 지표")
            c1, c2, c3 = st.columns(3)
            c1.metric("가짜뉴스 확률", f"{final_prob}%", delta=f"{total_score-50}")
            c2.metric("판정", "🚨 위험" if final_prob>60 else "🟢 안전" if final_prob<30 else "🟠 주의")
            c3.metric("AI 지능 레벨", f"{total_intelligence}", "+1")
            
            if is_official: st.success(f"🛡️ 공식 언론사({uploader})입니다.")
            if is_ai_content: st.warning(f"🤖 AI 콘텐츠 감지: {ai_reason}")
            if silent_penalty: st.error("🔇 침묵의 메아리: 자극적이나 근거가 부족합니다.")
            
            st.divider()
            col1, col2 = st.columns([1, 1.4])
            with col1:
                st.info(f"🎯 핀포인트 검색어: {refined_query}")
                st.write("**영상 요약**"); st.write(summary_text)
                st.table(pd.DataFrame([["기본 점수", 50], ["벡터 분석", t_impact+f_impact], ["뉴스 대조", news_score], ["침묵/불일치", silent_penalty+mismatch_penalty]], columns=["항목", "점수"]))
            with col2:
                colored_progress_bar("진실 유사도", t_sim, "green"); colored_progress_bar("거짓 유사도", f_sim, "red")
                st.write(f"**뉴스 검색 ({news_cnt}건)**"); st.table(pd.DataFrame(news_ev)) if news_ev else st.warning("관련 뉴스 없음")
                st.write("**댓글 분석**"); st.write(f"여론: {rel_msg}, 논란 키워드: {red_cnt}회")

        except Exception as e: st.error(f"분석 중 오류: {e}")

st.title("⚖️ Triple-Evidence Intelligence Forensic v48.0")
url = st.text_input("🔗 유튜브 URL 입력")
if st.button("🚀 분석 시작") and url: run_forensic_main(url)

st.divider()
st.subheader("🗂️ 학습 데이터 관리 (Cloud)")
try:
    df = pd.DataFrame(supabase.table("analysis_history").select("*").order("id", desc=True).execute().data)
    if not df.empty and st.session_state["is_admin"]:
        ed = st.data_editor(df, num_rows="dynamic", key="editor")
        if st.button("삭제 적용"):
            st.warning("기능 구현 중 (직접 DB 관리 권장)") 
    elif not df.empty: st.dataframe(df)
    else: st.info("데이터 없음")
except: pass
