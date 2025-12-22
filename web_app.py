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
st.set_page_config(page_title="Fact-Check Center v47.1 (Revert)", layout="wide", page_icon="⚖️")

# 🌟 Secrets에서 키 가져오기
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except:
    st.error("❌ 필수 키(API Key, DB Key, Password)가 설정되지 않았습니다.")
    st.stop()

# 🌟 Supabase 연결
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [관리자 인증 로직 (Form 사용)] ---
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

with st.sidebar:
    st.header("🛡️ 관리자 메뉴")
    
    with st.form("login_form"):
        password_input = st.text_input("관리자 비밀번호", type="password")
        submit_button = st.form_submit_button("로그인")
        
        if submit_button:
            if password_input == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True
                st.rerun()
            else:
                st.session_state["is_admin"] = False
                st.error("비밀번호가 일치하지 않습니다.")

    if st.session_state["is_admin"]:
        st.success("✅ 관리자 인증됨 (삭제 권한 보유)")
        if st.button("로그아웃"):
            st.session_state["is_admin"] = False
            st.rerun()
    else:
        st.info("데이터 삭제는 관리자만 가능합니다.")

# --- [상수 설정] ---
WEIGHT_NEWS_DEFAULT = 45        
WEIGHT_VECTOR = 35      
WEIGHT_CONTENT = 15     
WEIGHT_SENTIMENT_DEFAULT = 10   
PENALTY_ABUSE = 20      
PENALTY_MISMATCH = 30
PENALTY_NO_FACT = 25
PENALTY_SILENT_ECHO = 40   

# 핵심 상태어 사전
VITAL_KEYWORDS = [
    '위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '쓰러져', 
    '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', 
    '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '못넘긴다'
]

# VIP 인물 사전
VIP_ENTITIES = [
    '윤석열', '대통령', '이재명', '한동훈', '김건희', '문재인', '박근혜', '이명박',
    '트럼프', '바이든', '푸틴', '젤렌스키', '시진핑', '정은', 
    '이준석', '조국', '추미애', '홍준표', '유승민', '안철수',
    '손흥민', '이강인', '김민재', '류현진', '재용', '정의선', '최태원'
]

OFFICIAL_CHANNELS = [
    'MBC', 'KBS', 'SBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'YONHAP', 
    'NEWS', '뉴스', '채널A', 'TV조선', '연합뉴스', '한겨레', '경향', '조선', '중앙', '동아'
]

STATIC_TRUTH_CORPUS = [
    "박나래 위장전입 의혹 무혐의 수사 종결 공식 발표",
    "임영웅 콘서트 암표 소속사 강력 법적 대응 공지",
    "정희원 교수 저속노화 스토킹 피해 호소 언론 보도",
    "대전 충남 행정 통합 논의 지자체 공식 협의",
    "국회의원 선거 출마 공식 선언 기자회견",
    "강훈식 의원 충남지사 출마설 보도"
]
STATIC_FAKE_CORPUS = [
    "충격 폭로 경악 그 실체는?",
    "긴급 속보 알고보니 ㄷㄷ 소름 돋는 진실",
    "이재명 한동훈 충격 발언 논란",
    "결국 구속 영장 발부 눈물 바다",
    "방송 불가 판정 받은 영상 유출",
    "꿈속 계시 하나님 말씀 예언",
    "사형 선고 집행 확정",
    "건강 악화 위독설 응급실"
]

# --- [벡터 엔진] ---
class VectorEngine:
    def __init__(self):
        self.vocab = set()
        self.truth_vectors = []
        self.fake_vectors = []
    def tokenize(self, text):
        words = re.findall(r'[가-힣]{2,}', text)
        return [w for w in words]
    def build_vocabulary(self, corpus):
        for text in corpus:
            tokens = self.tokenize(text)
            self.vocab.update(tokens)
        self.vocab = sorted(list(self.vocab))
    def text_to_vector(self, text):
        tokens = self.tokenize(text)
        token_counts = Counter(tokens)
        vector = []
        for word in self.vocab: vector.append(token_counts[word])
        return vector
    def cosine_similarity(self, vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 == 0 or magnitude2 == 0: return 0.0
        return dot_product / (magnitude1 * magnitude2)
    def train(self, truth_corpus, fake_corpus):
        self.build_vocabulary(truth_corpus + fake_corpus)
        self.truth_vectors = [self.text_to_vector(t) for t in truth_corpus]
        self.fake_vectors = [self.text_to_vector(t) for t in fake_corpus]
    def analyze_position(self, query):
        query_vec = self.text_to_vector(query)
        max_truth_sim = 0
        for tv in self.truth_vectors:
            sim = self.cosine_similarity(query_vec, tv)
            if sim > max_truth_sim: max_truth_sim = sim
        max_fake_sim = 0
        for fv in self.fake_vectors:
            sim = self.cosine_similarity(query_vec, fv)
            if sim > max_fake_sim: max_fake_sim = sim
        return max_truth_sim, max_fake_sim

vector_engine = VectorEngine()

# --- [DB 함수] ---
def save_analysis(channel, title, prob, url, keywords):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {
        "channel_name": channel,
        "video_title": title,
        "fake_prob": prob,
        "analysis_date": now,
        "video_url": url,
        "keywords": keywords
    }
    try:
        supabase.table("analysis_history").insert(data).execute()
    except: pass

def train_dynamic_vector_engine():
    try:
        response_truth = supabase.table("analysis_history").select("video_title").lt("fake_prob", 30).execute()
        dynamic_truth = [row['video_title'] for row in response_truth.data]
        
        response_fake = supabase.table("analysis_history").select("video_title").gt("fake_prob", 70).execute()
        dynamic_fake = [row['video_title'] for row in response_fake.data]
    except:
        dynamic_truth, dynamic_fake = [], []
    
    final_truth = STATIC_TRUTH_CORPUS + dynamic_truth
    final_fake = STATIC_FAKE_CORPUS + dynamic_fake
    
    vector_engine.train(final_truth, final_fake)
    return len(final_truth) + len(final_fake)

# --- [UI Helper Functions] ---
def colored_progress_bar(label, percent, color):
    st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span style="font-size: 13px; font-weight: 600; color: #555;">{label}</span>
                <span style="font-size: 13px; font-weight: 700; color: {color};">{round(percent * 100, 1)}%</span>
            </div>
            <div style="background-color: #eee; border-radius: 5px; height: 8px; width: 100%;">
                <div style="background-color: {color}; height: 8px; width: {percent * 100}%; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_score_breakdown(data_list):
    style = """
    <style>
        table.score-table { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; font-family: sans-serif; font-size: 14px; margin-top: 10px;}
        table.score-table th { background-color: #f8f9fa; color: #495057; font-weight: bold; padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; }
        table.score-table td { padding: 12px 15px; border-bottom: 1px solid #f0f0f0; color: #333; }
        table.score-table tr:last-child td { border-bottom: none; }
        .badge { padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-align: center; min-width: 45px; }
        .badge-danger { background-color: #ffebee; color: #d32f2f; }
        .badge-success { background-color: #e8f5e9; color: #2e7d32; }
        .badge-neutral { background-color: #f5f5f5; color: #757575; border: 1px solid #e0e0e0; }
    </style>
    """
    rows = ""
    for item, score, note in data_list:
        try:
            score_num = int(score)
            if score_num > 0: badge = f'<span class="badge badge-danger">+{score_num}</span>'
            elif score_num < 0: badge = f'<span class="badge badge-success">{score_num}</span>'
            else: badge = f'<span class="badge badge-neutral">0</span>'
        except: badge = f'<span class="badge badge-neutral">{score}</span>'
        rows += f"<tr><td>{item}<br><span style='color:#888; font-size:11px;'>{note}</span></td><td style='text-align: right;'>{badge}</td></tr>"

    st.markdown(f"{style}<table class='score-table'><thead><tr><th>분석 항목 (Silent Echo Protocol)</th><th style='text-align: right;'>변동</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def witty_loading_sequence(count):
    messages = [
        f"🧠 [Intelligence Level: {count}] 누적 지식 로드 중...",
        "🔄 '주어(Modifier)' + '핵심어(Head)' 역방향 결합(Back-Merge) 중...",
        "🎯 문맥을 통합하여 완벽한 검색어(Contextual Query) 생성...",
        "🚀 위성이 유튜브 본사 상공을 지나가는 중..."
    ]
    with st.status("🕵️ Context Merger v46.0 가동 중...", expanded=True) as status:
        for msg in messages:
            st.write(msg)
            time.sleep(0.4)
        st.write("✅ 분석 준비 완료!")
        status.update(label="분석 완료!", state="complete", expanded=False)

# --- [Logic Functions] ---
def extract_nouns(text):
    noise = ['충격', '경악', '실체', '난리', '공개', '반응', '명단', '동영상', '사진', '집안', '속보', '단독', '결국', 'MBC', '뉴스', '이미지', '너무', '다른', '알고보니', 'ㄷㄷ', '진짜', '정말', '영상', '사람', '생각', '오늘밤', '오늘', '내일', '지금', '못넘긴다', '넘긴다', '이유', '왜', '안']
    nouns = re.findall(r'[가-힣]{2,}', text)
    return list(dict.fromkeys([n for n in nouns if n not in noise]))

# 🌟 핀포인트 쿼리 생성 (Chunking + SOV Back-Merge)
def generate_pinpoint_query(title, hashtags):
    clean_text = title + " " + " ".join([h.replace("#", "") for h in hashtags])
    words = clean_text.split()
    
    subject_chunk = ""
    object_word = ""
    vital_word = ""
    
    for vital in VITAL_KEYWORDS:
        if vital in clean_text:
            vital_word = vital
            break
            
    for i, word in enumerate(words):
        match = re.match(r'([가-힣A-Za-z0-9]+)(은|는|이|가|을|를|에|에게|로서|로)', word)
        
        if match:
            noun = match.group(1)
            josa = match.group(2)
            
            if noun in ['오늘밤', '지금', '이유', '결국']: continue

            # 주어 찾기 + 역방향 결합
            if not subject_chunk and josa in ['은', '는', '이', '가']:
                prev_noun = ""
                if i > 0:
                    prev_word = words[i-1]
                    if re.fullmatch(r'[가-힣A-Za-z0-9]+', prev_word):
                        if prev_word not in VITAL_KEYWORDS and prev_word not in ['충격', '속보']:
                            prev_noun = prev_word
                
                if prev_noun:
                    subject_chunk = f"{prev_noun} {noun}"
                else:
                    subject_chunk = noun
            
            # 목적어 찾기
            elif not object_word and josa in ['을', '를', '에', '에게', '로']:
                if noun not in VITAL_KEYWORDS and noun not in subject_chunk:
                    object_word = noun
    
    if not subject_chunk:
        nouns = extract_nouns(title)
        return " ".join(nouns[:3])
    
    query_parts = []
    if subject_chunk: query_parts.append(subject_chunk)
    if object_word: query_parts.append(object_word)
    if vital_word: query_parts.append(vital_word)
    
    return " ".join(query_parts)

def summarize_transcript(text, max_sentences=3):
    if not text or len(text) < 50:
        return "⚠️ 요약할 자막 내용이 충분하지 않습니다."
    sentences = re.split(r'(?<=[.?!])\s+', text)
    if len(sentences) <= max_sentences: return text
    nouns = re.findall(r'[가-힣]{2,}', text)
    word_freq = Counter(nouns)
    ranked_sentences = []
    for i, sent in enumerate(sentences):
        sent_nouns = re.findall(r'[가-힣]{2,}', sent)
        if not sent_nouns: continue
        score = sum(word_freq[w] for w in sent_nouns)
        if 10 < len(sent) < 150:
            ranked_sentences.append((i, sent, score / len(sent_nouns)))
    top_sentences = sorted(ranked_sentences, key=lambda x: x[2], reverse=True)[:max_sentences]
    top_sentences.sort(key=lambda x: x[0])
    summary = " ".join([s[1] for s in top_sentences])
    return f"📌 **핵심 요약**: {summary}"

def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text()

def detect_ai_content(info):
    is_ai = False
    reasons = []
    ai_keywords = ['ai', 'artificial intelligence', 'chatgpt', 'midjourney', 'sora', 'deepfake', 'synthetic', '인공지능', '딥페이크', '가상인간', '버추얼', 'gpt']
    text_to_check = (info.get('title', '') + " " + info.get('description', '') + " " + " ".join(info.get('tags', []))).lower()
    for kw in ai_keywords:
        if re.search(r'\b{}\b'.format(re.escape(kw)), text_to_check):
            is_ai = True
            reasons.append(f"키워드 감지: {kw}")
            break
    return is_ai, ", ".join(reasons)

def check_is_official(channel_name):
    norm_name = channel_name.upper().replace(" ", "")
    for official in OFFICIAL_CHANNELS:
        if official in norm_name: return True
    return False

def extract_nouns_list(text):
    noise = ['충격', '경악', '실체', '난리', '공개', '반응', '명단', '동영상', '사진', '집안', '속보', '단독', '결국', 'MBC', '뉴스', '이미지', '너무', '다른', '알고보니', 'ㄷㄷ', '진짜', '정말', '영상', '사람', '생각', '최고', '응원', '화이팅', '사랑']
    nouns = re.findall(r'[가-힣]{2,}', text)
    return [n for n in nouns if n not in noise]

def count_sensational_words(text):
    triggers = ['충격', '경악', '실체', '폭로', '난리', '속보', '긴급', '소름', 'ㄷㄷ', '진짜', '결국', '계시', '예언', '위독', '사망', '중태']
    count = 0
    for w in triggers: count += text.count(w)
    return count

def check_tag_abuse(title, hashtags, channel_name):
    is_official = check_is_official(channel_name)
    if is_official: return 0, "공식 채널 면제"
    if not hashtags: return 0, "해시태그 없음"
    title_nouns = extract_nouns(title)
    tag_nouns = set()
    for t in hashtags:
        val = t.replace("#", "").split(":")[-1].strip()
        tag_nouns.add(val)
    if len(tag_nouns) < 2: return 0, "양호"
    # 🌟 [Fix] set 변환
    intersection = set(title_nouns).intersection(tag_nouns)
    if not intersection: return PENALTY_ABUSE, "🚨 심각 (불일치)"
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
            lines = response.text.splitlines()
            clean_lines = []
            seen = set()
            for line in lines:
                line = line.strip()
                if '-->' in line or line == 'WEBVTT' or not line: continue
                line = re.sub(r'<[^>]+>', '', line)
                if line and line not in seen:
                    clean_lines.append(line)
                    seen.add(line)
            return " ".join(clean_lines), "✅ 실제 자막 수집 성공"
    except: pass
    return None, "자막 다운로드 실패"

def fetch_comments_via_api(video_id):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {'part': 'snippet', 'videoId': video_id, 'key': YOUTUBE_API_KEY, 'maxResults': 100, 'order': 'relevance'}
    try:
        res = requests.get(url, params=params)
        if res.status_code == 200:
            data = res.json()
            items = data.get('items', [])
            comment_data = []
            for item in items:
                snippet = item['snippet']['topLevelComment']['snippet']
                text = snippet['textDisplay']
                likes = snippet.get('likeCount', 0)
                comment_data.append({'text': text, 'likes': likes})
            comment_data.sort(key=lambda x: x['likes'], reverse=True)
            top_comments = [c['text'] for c in comment_data[:50]]
            return top_comments, f"✅ API 수집 성공 (Top {len(top_comments)} by Likes)"
        elif res.status_code == 403: return [], "⚠️ API 권한 오류"
        elif res.status_code == 404: return [], "⚠️ 댓글 사용 중지됨"
        else: return [], f"⚠️ API 오류 ({res.status_code})"
    except Exception as e: return [], f"❌ API 통신 실패"

def calculate_dual_match(news_item, query_nouns, transcript):
    if not news_item or not transcript: return 0, 0, 0
    news_title = news_item.get('title', '')
    title_nouns = extract_nouns(news_title)
    
    # 🌟 [Fix] set 변환
    intersection = len(set(query_nouns).intersection(set(title_nouns)))
    title_score = 1.0 if intersection >= 2 else (0.5 if intersection == 1 else 0)
    
    news_desc = news_item.get('desc', '')
    desc_nouns = extract_nouns(news_desc)
    
    if not desc_nouns:
        content_score = 0
    else:
        match_count = 0
        for noun in desc_nouns:
            if noun in transcript:
                match_count += 1
        content_ratio = match_count / len(desc_nouns)
        content_score = 1.0 if content_ratio >= 0.3 else (0.5 if content_ratio >= 0.15 else 0)
    total_score = (title_score * 0.3) + (content_score * 0.7)
    return int(total_score * 100), int(title_score * 100), int(content_score * 100)

def analyze_comment_relevance(comments, context_text):
    if not comments: return [], 0, "분석 불가"
    all_comments_text = " ".join(comments)
    comment_nouns = extract_nouns_list(all_comments_text)
    if not comment_nouns: return [], 0, "유효 키워드 없음"
    top_keywords = Counter(comment_nouns).most_common(5)
    top_words_only = [word for word, cnt in top_keywords]
    context_nouns = extract_nouns(context_text)
    match_count = 0
    context_set = set(context_nouns)
    for word in top_words_only:
        if word in context_set: match_count += 1
    relevance_score = int((match_count / len(top_keywords)) * 100)
    if relevance_score >= 60: relevance_msg = "✅ 주제 집중 토론형"
    elif relevance_score >= 20: relevance_msg = "⚠️ 일부 관련 / 잡담 혼재"
    else: relevance_msg = "❌ 무관한 딴소리 / 맹목적 지지"
    formatted_keywords = [f"{w}({c})" for w, c in top_keywords]
    return formatted_keywords, relevance_score, relevance_msg

def check_red_flags(comments):
    red_flag_keywords = ['가짜뉴스', '가짜 뉴스', '주작', '사기', '거짓말', '허위', '구라', '합성', '선동', '소설']
    count = 0
    detected = []
    for c in comments:
        for k in red_flag_keywords:
            if k in c:
                count += 1
                detected.append(k)
    return count, list(set(detected))

# --- [8. 실행부] ---
def run_forensic_main(url):
    total_intelligence = train_dynamic_vector_engine()
    witty_loading_sequence(total_intelligence)
    
    video_id = None
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if match: video_id = match.group(1)

    ydl_opts = {
        'quiet': True, 'skip_download': True, 'get_comments': False,
        'writesubtitles': True, 'writeautomaticsub': True,
        'subtitleslangs': ['ko'],
        'extractor_args': {'youtube': {'skip': ['dash', 'hls']}}
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', '제목 없음')
            uploader = info.get('uploader', '알 수 없음')
            all_hashtags = info.get('tags', [])
            description = info.get('description', '')
            
            is_official = check_is_official(uploader)
            is_ai_content, ai_reason = detect_ai_content(info)
            
            current_weight_news = WEIGHT_NEWS_DEFAULT
            current_weight_vector = WEIGHT_VECTOR
            current_weight_sentiment = WEIGHT_SENTIMENT_DEFAULT
            
            if is_ai_content:
                current_weight_news = 70  
                current_weight_vector = 10 
            
            refined_query = generate_pinpoint_query(title, all_hashtags)
            hashtag_display = ", ".join([f"#{t}" for t in all_hashtags]) if all_hashtags else "해시태그 없음"
            abuse_score, abuse_status = check_tag_abuse(title, all_hashtags, uploader)
            
            transcript_text, transcript_status = fetch_real_transcript(info)
            analysis_text = transcript_text if transcript_text else description
            
            summary_text = summarize_transcript(analysis_text)
            
            agitation_count = count_sensational_words(analysis_text + title)
            agitation_level = "높음 (위험)" if agitation_count > 3 else "보통" if agitation_count > 0 else "낮음 (안전)"
            
            t_sim, f_sim = vector_engine.analyze_position(refined_query + " " + title)
            t_impact = int(t_sim * current_weight_vector) * -1 
            f_impact = int(f_sim * current_weight_vector)

            max_news_sim, news_ev, news_collected_cnt, news_used_cnt = 0, [], 0, 0
            search_q = refined_query 
            max_dual_score = 0
            best_veracity_display = "0%"
            best_match_content_score = 0
            
            try:
                rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(search_q)}&hl=ko&gl=KR"
                r = requests.get(rss_url, timeout=5)
                root = ET.fromstring(r.content)
                items = root.findall('.//item')
                
                if not items:
                    fallback_q = " ".join(search_q.split()[:2])
                    rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(fallback_q)}&hl=ko&gl=KR"
                    r = requests.get(rss_url, timeout=5)
                    root = ET.fromstring(r.content)
                    items = root.findall('.//item')
                    if items: search_q = fallback_q
                
                news_collected_cnt = len(items)
                news_used_cnt = min(len(items), 3)
                
                for i, item in enumerate(items[:3]):
                    try:
                        nt = item.find('title').text
                        raw_desc = item.find('description').text if item.find('description') is not None else ""
                        clean_desc = clean_html(raw_desc)
                        news_item_dict = {'title': nt, 'desc': clean_desc}
                        query_nouns = extract_nouns(search_q)
                        
                        total, t_sc, c_sc = calculate_dual_match(news_item_dict, query_nouns, analysis_text)
                        
                        if total > max_dual_score: 
                            max_dual_score = total
                            best_veracity_display = f"{c_sc}% (Content Match)"
                            best_match_content_score = c_sc
                        
                        news_ev.append({"뉴스 제목": nt, "최종 일치도": f"{total}%", "상세": f"(제목:{t_sc}%, 내용:{c_sc}%)"})
                    except: continue
            except: pass
            
            total_text_len = len(analysis_text.split())
            analyzed_nouns = len(extract_nouns(analysis_text))
            info_density = round((analyzed_nouns / total_text_len) * 100, 1) if total_text_len > 0 else 0
            
            comments_list, comments_status = fetch_comments_via_api(video_id)
            cmts_collected_cnt = len(comments_list)
            used_comments = comments_list
            top_keywords, relevance_score, relevance_msg = analyze_comment_relevance(used_comments, title + " " + analysis_text)
            
            red_flag_count, red_flag_words = check_red_flags(used_comments)
            is_controversial = False
            
            if red_flag_count > 0:
                is_controversial = True
                current_weight_news = 65
                current_weight_sentiment = 0
            
            final_sim_ratio = max_dual_score / 100.0
            adjusted_ratio = math.pow(final_sim_ratio, 2)
            
            silent_echo_penalty = 0
            is_effective_silence = (news_collected_cnt == 0) or (news_collected_cnt > 0 and max_dual_score < 20)
            
            if is_effective_silence:
                if agitation_count >= 3: 
                    silent_echo_penalty = PENALTY_SILENT_ECHO 
                    t_impact *= 2
                    f_impact *= 2
                news_safety_score = 0
                news_note = "No Relevant News (Effective Silent Echo)"
                
            elif is_controversial:
                if max_dual_score < 60:
                    news_safety_score = PENALTY_NO_FACT 
                    news_note = "Penalty: Unverified despite Controversy"
                else:
                    news_safety_score = int(adjusted_ratio * current_weight_news) * -1
                    news_note = f"Max -{current_weight_news} (Verified Conflict)"
            else:
                news_safety_score = int(adjusted_ratio * current_weight_news) * -1
                if 0 < best_match_content_score < 70:
                    news_safety_score = int(news_safety_score * 0.5)
                news_note = f"Max -{current_weight_news} (Standard)"

            is_misleading = (news_collected_cnt > 0) and (max_dual_score < 20)
            mismatch_penalty = 0
            if is_official:
                is_misleading = False
                news_safety_score = -50
                mismatch_penalty = 0
                news_note = "Official Channel Bonus"
            elif is_misleading:
                news_safety_score = 0
                mismatch_penalty = PENALTY_MISMATCH 
                news_note = "Score Voided (Mismatch)"

            sentiment_score = 0
            if cmts_collected_cnt > 0 and not is_controversial:
                s_counts = Counter([s for c in used_comments for s in ['가짜', '낚시', '조작', '선동'] if s in c])
                neg_ratio = min(1.0, sum(s_counts.values()) / len(used_comments)) if used_comments else 0
                sentiment_score = int(neg_ratio * current_weight_sentiment)

            clickbait_words = ['충격', '경악', '실체', '난리', '결국', '폭로']
            clickbait_score = 10 if any(w in title for w in clickbait_words) else -5
            
            base_score = 50
            total_score = base_score + t_impact + f_impact + news_safety_score + sentiment_score + clickbait_score + abuse_score + mismatch_penalty + silent_echo_penalty
            final_prob = max(5, min(99, total_score))
            
            save_analysis(uploader, title, final_prob, url, refined_query)

            # --- UI ---
            st.subheader("🕵️ 핵심 분석 지표 (Key Indicators)")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("최종 가짜뉴스 확률", f"{final_prob}%", delta=f"{total_score - base_score}")
            with col_b:
                icon = "🟢" if final_prob < 30 else "🔴" if final_prob > 60 else "🟠"
                verdict = "매우 안전" if final_prob < 30 else "위험 감지" if final_prob > 60 else "주의 요망"
                st.metric("종합 AI 판정", f"{icon} {verdict}")
            with col_c:
                st.metric("AI Intelligence Level", f"{total_intelligence} Knowledge Nodes", delta="+1 Added")

            if is_ai_content:
                st.warning(f"🤖 **AI 생성 콘텐츠 감지됨**: {ai_reason}")
            if is_official:
                st.success(f"🛡️ **공식 언론사 채널({uploader})입니다.**")
            
            if silent_echo_penalty > 0:
                st.error(f"🔇 **침묵의 메아리(Silent Echo) 경고**: 자극적인 주장이나 의혹을 제기하고 있으나, 이를 뒷받침할 제도권 언론 보도가 전무합니다. 가짜뉴스일 확률이 매우 높습니다.")

            st.divider()
            col1, col2 = st.columns([1, 1.4])
            with col1:
                st.write("**[영상 상세 정보]**")
                meta_df = pd.DataFrame({
                    "항목": ["영상 제목", "채널명", "카테고리", "타입", "업로드일", "조회수", "해시태그"],
                    "내용": [title, uploader, info.get('categories', ['미분류'])[0], "쇼츠" if "shorts" in url else "일반", info.get('upload_date', 'N/A'), f"{info.get('view_count', 0):,}회", hashtag_display]
                })
                st.table(meta_df)
                
                st.info(f"🎯 **핀포인트 뉴스 검색어**: {search_q}")
                
                with st.container(border=True):
                    st.markdown("📝 **영상 내용 요약 (AI Abstract)**")
                    st.caption("자막 데이터를 분석하여 핵심 문장 3개를 추출한 결과입니다.")
                    st.write(summary_text)
                
                st.write("**[Score Breakdown]**")
                score_data = [
                    ["기본 위험도", 50, "Base Score"],
                    ["진실 맥락 보너스 (벡터)", t_impact, f"Dynamic Weight: x{2 if silent_echo_penalty else 1}"],
                    ["가짜 패턴 가점 (벡터)", f_impact, f"Dynamic Weight: x{2 if silent_echo_penalty else 1}"],
                    ["뉴스 교차 대조 (Dual)", news_safety_score, news_note],
                    ["침묵의 메아리 (No News)", f"+{silent_echo_penalty}" if silent_echo_penalty else "0", "Penalty for No/Irrelevant News"],
                    ["여론/제목/자막 가감", sentiment_score + clickbait_score, f"Sent: {sentiment_score}"],
                    ["내용 불일치 기만", mismatch_penalty, f"Penalty +{PENALTY_MISMATCH} (Title Baiting)"],
                    ["해시태그 어뷰징", f"+{abuse_score}" if abuse_score > 0 else "0 (면제/정상)", f"Penalty +{PENALTY_ABUSE}"]
                ]
                render_score_breakdown(score_data)

            with col2:
                st.subheader("📊 5대 정밀 분석 증거")
                
                st.markdown("**[증거 0] Semantic Vector Space (진실/거짓 분포)**")
                st.caption(f"💡 Intelligence Level {total_intelligence} 기반 분석")
                colored_progress_bar("✅ 진실 영역 근접도", t_sim, "#2ecc71")
                colored_progress_bar("🚨 거짓 영역 근접도", f_sim, "#e74c3c")
                
                st.write("---")
                
                st.markdown(f"**[증거 1] 뉴스 교차 대조 (Query: {search_q})**")
                st.caption(f"📡 수집: **{news_collected_cnt}건** | 🧪 분석: **상위 {news_used_cnt}건**")
                if news_ev: st.table(pd.DataFrame(news_ev))
                else: st.warning("🔍 관련 뉴스를 찾을 수 없습니다. (Silent Echo Risk Increased)")
                
                st.markdown("**[증거 2] 시청자 여론 심층 분석**")
                st.caption(f"💬 상태: **{comments_status}**")
                if cmts_collected_cnt > 0:
                    opinion_df = pd.DataFrame([
                        ["최다 빈출 키워드", ", ".join(top_keywords)],
                        ["논란 감지 여부", f"{'⚠️ 감지됨' if is_controversial else '✅ 안정적'} ({red_flag_count}회)"],
                        ["주제 일치도", f"{relevance_score}% ({relevance_msg})"]
                    ], columns=["항목", "내용"])
                    st.table(opinion_df)
                else: st.warning("⚠️ 댓글 수집 불가.")
                
                st.markdown("**[증거 3] 자막 세만틱 심층 대조**")
                st.caption(f"📝 **{transcript_status}** | 📚 전체 단어: **{total_text_len}개**")
                semantic_df = pd.DataFrame([
                    ["제목 낚시어", f"{', '.join([w for w in clickbait_words if w in title]) if any(w in title for w in clickbait_words) else '없음'}"],
                    ["정보 밀도 (명사/전체)", f"{info_density}% ({'높음' if info_density > 20 else '낮음'})"],
                    ["선동성 지수", f"{agitation_level} ({agitation_count}회)"],
                    ["기사-영상 일치도", f"{max_dual_score}% (종합) / {best_veracity_display}"]
                ], columns=["분석 항목", "판정 결과"])
                st.table(semantic_df)
                
                st.markdown("**[증거 4] AI 최종 분석 판단**")
                result_text = f"현재 분석된 종합 점수는 {final_prob}점입니다. "
                if is_official:
                    result_text += "🛡️ **공식 언론사 채널**로 확인되어 신뢰할 수 있는 정보입니다. "
                elif silent_echo_penalty > 0:
                    result_text += "🔇 **자극적 주장을 뒷받침할 언론 보도가 없거나 관련성이 낮습니다(Silent Echo).** 가짜뉴스일 확률이 매우 높습니다. "
                elif is_controversial and max_dual_score < 60:
                    result_text += "🚨 **경고: 영상에 대한 논란(가짜뉴스 의혹)이 있으나, 이를 뒷받침할 명확한 뉴스 보도가 부족합니다(Fact Deficit).** 위험도가 상향 조정되었습니다. "
                elif is_misleading:
                    result_text += "🚨 **경고: 제목과 내용이 불일치하거나, 실제 보도 내용과 다른 '낚시성 영상'으로 판단됩니다.** "
                
                if final_prob < 30 and not is_misleading:
                    result_text += "안전한 콘텐츠로 판단됩니다."
                elif final_prob > 60:
                    result_text += "주의가 필요한 콘텐츠입니다."
                st.success(f"🔍 {result_text}")
                
                if final_prob < 30 or final_prob > 70:
                    st.toast(f"🤖 AI가 이 결과를 학습했습니다!", icon="🧠")

        except Exception as e: st.error(f"오류: {e}")

# --- [9. 실행부] ---
st.title("⚖️ Triple-Evidence Intelligence Forensic v47.1")

with st.container(border=True):
    st.markdown("""
    ### 🛡️ 법적 고지 및 책임 한계 (Disclaimer)
    본 서비스는 **인공지능(AI) 및 알고리즘 기반**으로 영상의 신뢰도를 분석하는 보조 도구입니다. 
    * **최종 판단의 주체:** 정보의 진위 여부에 대한 최종적인 판단과 그에 따른 책임은 **사용자 본인**에게 있습니다.
    """)
    agree = st.checkbox("위 내용을 확인하였으며, 이에 동의합니다. (동의 시 분석 버튼 활성화)")

url_input = st.text_input("🔗 분석할 유튜브 URL")

if st.button("🚀 정밀 분석 시작", use_container_width=True, disabled=not agree):
    if url_input: run_forensic_main(url_input)
    else: st.warning("URL을 입력해주세요.")

st.divider()
st.subheader("🗂️ 학습 데이터 관리 (Cloud Knowledge Base)")
st.caption("☁️ 이 데이터는 서버가 재부팅되어도 사라지지 않는 영구적인 집단지성 데이터입니다.")

try:
    response = supabase.table("analysis_history").select("*").order("id", desc=True).execute()
    df = pd.DataFrame(response.data)
except:
    df = pd.DataFrame()

if not df.empty:
    df['Delete'] = False
    cols = ['Delete', 'id', 'analysis_date', 'video_title', 'fake_prob', 'keywords']
    df = df[cols]

    # 🌟 관리자 권한 확인 (삭제 버튼 제어)
    if st.session_state.get("is_admin", False):
        edited_df = st.data_editor(
            df,
            column_config={
                "Delete": st.column_config.CheckboxColumn("선택 삭제", default=False),
                "fake_prob": st.column_config.ProgressColumn("가짜 확률", format="%d%%", min_value=0, max_value=100),
            },
            disabled=["id", "analysis_date", "video_title", "keywords"],
            hide_index=True,
            use_container_width=True
        )

        to_delete = edited_df[edited_df.Delete]
        if not to_delete.empty:
            if st.button(f"🗑️ 선택한 {len(to_delete)}건의 기록 영구 삭제", type="primary"):
                try:
                    for index, row in to_delete.iterrows():
                        supabase.table("analysis_history").delete().eq("id", row['id']).execute()
                    
                    st.success("✅ 클라우드 DB에서 데이터가 삭제되었습니다.")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"삭제 중 오류 발생: {e}")
    else:
        # 일반 사용자는 읽기 전용
        st.dataframe(df.drop(columns=['Delete']), hide_index=True, use_container_width=True)
        st.info("🔒 데이터 삭제 권한이 없습니다. (관리자 로그인 필요)")
else:
    st.info("☁️ 클라우드 DB에 저장된 분석 기록이 없습니다.")
