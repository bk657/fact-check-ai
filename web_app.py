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
st.set_page_config(page_title="Fact-Check Center v47.1 (Secure)", layout="wide", page_icon="⚖️")

# 🌟 Secrets에서 키 가져오기
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"] # 관리자 비번 가져오기
except:
    st.error("❌ 필수 키(API Key, DB Key, Password)가 설정되지 않았습니다.")
    st.stop()

# 🌟 Supabase 연결
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [관리자 인증 로직] ---
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

def check_password():
    """비밀번호 확인 함수"""
    if st.session_state["password_input"] == ADMIN_PASSWORD:
        st.session_state["is_admin"] = True
    else:
        st.session_state["is_admin"] = False

# 사이드바에 로그인 창 배치
with st.sidebar:
    st.header("🛡️ 관리자 메뉴")
    st.text_input(
        "관리자 비밀번호", 
        type="password", 
        key="password_input", 
        on_change=check_password
    )
    if st.session_state["is_admin"]:
        st.success("✅ 관리자 인증됨")
    else:
        st.info("데이터 삭제 권한은 관리자에게만 있습니다.")

# --- [상수 및 클래스 정의] ---
WEIGHT_NEWS_DEFAULT = 45       
WEIGHT_VECTOR = 35     
WEIGHT_CONTENT = 15    
WEIGHT_SENTIMENT_DEFAULT = 10  
PENALTY_ABUSE = 20     
PENALTY_MISMATCH = 30
PENALTY_NO_FACT = 25
PENALTY_SILENT_ECHO = 40  

VITAL_KEYWORDS = [
    '위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '쓰러져', 
    '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', 
    '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '못넘긴다'
]

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

# --- [DB Functions] ---
def save_analysis(channel, title, prob, url, keywords):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {"channel_name": channel, "video_title": title, "fake_prob": prob, "analysis_date": now, "video_url": url, "keywords": keywords}
    try: supabase.table("analysis_history").insert(data).execute()
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

# --- [Helper Functions] ---
def colored_progress_bar(label, percent, color):
    st.markdown(f"""<div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span style="font-size: 13px; font-weight: 600; color: #555;">{label}</span><span style="font-size: 13px; font-weight: 700; color: {color};">{round(percent * 100, 1)}%</span></div><div style="background-color: #eee; border-radius: 5px; height: 8px; width: 100%;"><div style="background-color: {color}; height: 8px; width: {percent * 100}%; border-radius: 5px;"></div></div></div>""", unsafe_allow_html=True)

def render_score_breakdown(data_list):
    style = """<style>table.score-table { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; font-family: sans-serif; font-size: 14px; margin-top: 10px;} table.score-table th { background-color: #f8f9fa; color: #495057; font-weight: bold; padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; } table.score-table td { padding: 12px 15px; border-bottom: 1px solid #f0f0f0; color: #333; } table.score-table tr:last-child td { border-bottom: none; } .badge { padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-align: center; min-width: 45px; } .badge-danger { background-color: #ffebee; color: #d32f2f; } .badge-success { background-color: #e8f5e9; color: #2e7d32; } .badge-neutral { background-color: #f5f5f5; color: #757575; border: 1px solid #e0e0e0; }</style>"""
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
    messages = [f"🧠 [Intelligence Level: {count}] 누적 지식 로드 중...", "🔄 '주어(Modifier)' + '핵심어(Head)' 역방향 결합(Back-Merge) 중...", "🎯 문맥을 통합하여 완벽한 검색어(Contextual Query) 생성...", "🚀 위성이 유튜브 본사 상공을 지나가는 중..."]
    with st.status("🕵️ Context Merger v46.0 가동 중...", expanded=True) as status:
        for msg in messages: st.write(msg); time.sleep(0.4)
        st.write("✅ 분석 준비 완료!"); status.update(label="분석 완료!", state="complete", expanded=False)

def extract_nouns(text):
    noise = ['충격', '경악', '실체', '난리', '공개', '반응', '명단', '동영상', '사진', '집안', '속보', '단독', '결국', 'MBC', '뉴스', '이미지', '너무', '다른', '알고보니', 'ㄷㄷ', '진짜', '정말', '영상', '사람', '생각', '오늘밤', '오늘', '내일', '지금', '못넘긴다', '넘긴다', '이유', '왜', '안']
    nouns = re.findall(r'[가-힣]{2,}', text)
    return list(dict.fromkeys([n for n in nouns if n not in noise]))

def generate_pinpoint_query(title, hashtags):
    clean_text = title + " " + " ".join([h.replace("#", "") for h in hashtags])
    words = clean_text.split()
    subject_chunk, object_word, vital_word = "", "", ""
    
    for vital in VITAL_KEYWORDS:
        if vital in clean_text: vital_word = vital; break
            
    for i, word in enumerate(words):
        match = re.match(r'([가-힣A-Za-z0-9]+)(은|는|이|가|을|를|에|에게|로서|로)', word)
        if match:
            noun, josa = match.group(1), match.group(2)
            if noun in ['오늘밤', '지금', '이유', '결국']: continue
            if not subject_chunk and josa in ['은', '는', '이', '가']:
                prev_noun = ""
                if i > 0:
                    prev_word = words[i-1]
                    if re.fullmatch(r'[가-힣A-Za-z0-9]+', prev_word):
                        if prev_word not in VITAL_KEYWORDS and prev_word not in ['충격', '속보']: prev_noun = prev_word
                subject_chunk = f"{prev_noun} {noun}" if prev_noun else noun
            elif not object_word and josa in ['을', '를', '에', '에게', '로']:
                if noun not in VITAL_KEYWORDS and noun not in subject_chunk: object_word = noun
    
    if not subject_chunk:
        nouns = extract_nouns(title)
        return " ".join(nouns[:3])
    
    query_parts = []
    if subject_chunk: query_parts.append(subject_chunk)
    if object_word: query_parts.append(object_word)
    if vital_word: query_parts.append(vital_word)
    return " ".join(query_parts)

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
    all_comments_text = " ".join(comments); comment_nouns = extract_nouns_list(all_comments_text)
    if not comment_nouns: return [], 0, "유효 키워드 없음"
    top_keywords = Counter(comment_nouns).most_common(5)
    context_nouns = extract_nouns(context_text); match_count = 0; context_set = set(context_nouns)
    for word, cnt in top_keywords:
        if word in context_set: match_count += 1
    relevance_score = int((match_count / len(top_keywords)) * 100)
    if relevance_score >= 60: relevance_msg = "✅ 주제 집중 토론형"
    elif relevance_score >= 20: relevance_msg = "⚠️ 일부 관련 / 잡담 혼재"
    else: relevance_msg = "❌ 무관한 딴소리 / 맹목적 지지"
    return [f"{w}({c})" for w, c in top_keywords], relevance_score, relevance_msg

def check_red_flags(comments):
    red_flag_keywords = ['가짜뉴스', '가짜 뉴스', '주작', '사기', '거짓말', '허위', '구라', '합성', '선동', '소설']
    count = 0; detected = []
    for c in comments:
        for k in red_flag_keywords:
            if k in c: count += 1; detected.append(k)
    return count, list(set(detected))

# --- [Main Execution] ---
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
except: df = pd.DataFrame()

if not df.empty:
    df['Delete'] = False
    cols = ['Delete', 'id', 'analysis_date', 'video_title', 'fake_prob', 'keywords']
    df = df[cols]

    # 🌟 관리자만 삭제 가능하도록 UI 분기
    if st.session_state.get("is_admin", False):
        edited_df = st.data_editor(
            df,
            column_config={
                "Delete": st.column_config.CheckboxColumn("선택 삭제", default=False),
                "fake_prob": st.column_config.ProgressColumn("가짜 확률", format="%d%%", min_value=0, max_value=100),
            },
            disabled=["id", "analysis_date", "video_title", "keywords"],
            hide_index=True, use_container_width=True
        )
        to_delete = edited_df[edited_df.Delete]
        if not to_delete.empty:
            if st.button(f"🗑️ 선택한 {len(to_delete)}건의 기록 영구 삭제", type="primary"):
                try:
                    for index, row in to_delete.iterrows():
                        supabase.table("analysis_history").delete().eq("id", row['id']).execute()
                    st.success("✅ 클라우드 DB에서 데이터가 삭제되었습니다.")
                    time.sleep(1); st.rerun()
                except Exception as e: st.error(f"삭제 중 오류 발생: {e}")
    else:
        st.dataframe(df.drop(columns=['Delete']), hide_index=True, use_container_width=True)
        st.info("🔒 데이터 삭제 권한이 없습니다. (관리자 로그인 필요)")
else:
    st.info("☁️ 클라우드 DB에 저장된 분석 기록이 없습니다.")