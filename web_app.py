import streamlit as st
from supabase import create_client, Client
import re
import requests
import time
import random
import math
import google.generativeai as genai
from datetime import datetime
from collections import Counter
import yt_dlp
import pandas as pd
import altair as alt

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="Fact-Check Center v63.5 (Force Debug)", layout="wide", page_icon="⚖️")

# 🌟 Secrets 로드
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("❌ 필수 키(API Key, DB Key, Password)가 설정되지 않았습니다.")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [2. 상수 정의] ---
WEIGHT_NEWS_DEFAULT = 45; WEIGHT_VECTOR = 35; WEIGHT_CONTENT = 15; WEIGHT_SENTIMENT_DEFAULT = 10
PENALTY_ABUSE = 20; PENALTY_MISMATCH = 30; PENALTY_NO_FACT = 25; PENALTY_SILENT_ECHO = 40

VITAL_KEYWORDS = ['위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '내란', '간첩']
CRITICAL_STATE_KEYWORDS = ['별거', '이혼', '파경', '사망', '위독', '구속', '체포', '실형', '불화', '폭로', '충격', '논란', '중태', '심정지', '뇌사', '압수수색', '소환', '파산', '빚더미', '전과', '감옥', '간첩']
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']

STATIC_TRUTH_CORPUS = ["박나래 위장전입 무혐의", "임영웅 암표 대응", "정희원 저속노화", "대전 충남 통합", "선거 출마 선언"]
STATIC_FAKE_CORPUS = ["충격 폭로 경악", "긴급 속보 소름", "충격 발언 논란", "구속 영장 발부", "영상 유출", "계시 예언", "사형 집행", "위독설"]

# --- [3. NLP & Vector Engine] ---
class VectorEngine:
    def __init__(self):
        self.vocab = set()
        self.truth_vectors = []
        self.fake_vectors = []
    def tokenize(self, text): return re.findall(r'[가-힣]{2,}', text)
    def train(self, truth, fake):
        for t in truth + fake: self.vocab.update(self.tokenize(t))
        self.vocab = sorted(list(self.vocab))
        self.truth_vectors = [self.text_to_vector(t) for t in truth]
        self.fake_vectors = [self.text_to_vector(t) for t in fake]
    def text_to_vector(self, text):
        c = Counter(self.tokenize(text))
        return [c[w] for w in self.vocab]
    def cosine_similarity(self, v1, v2):
        dot = sum(a*b for a,b in zip(v1,v2))
        mag = math.sqrt(sum(a*a for a in v1)) * math.sqrt(sum(b*b for b in v2))
        return dot/mag if mag>0 else 0
    def analyze_position(self, query):
        qv = self.text_to_vector(query)
        mt = max([self.cosine_similarity(qv, v) for v in self.truth_vectors] or [0])
        mf = max([self.cosine_similarity(qv, v) for v in self.fake_vectors] or [0])
        return mt, mf

vector_engine = VectorEngine()

# --- [4. Gemini Logic (모델 순환 + 에러 노출)] ---
def get_gemini_search_keywords(title, transcript):
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # 1. 안전 설정 (최대 개방)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    # 2. 모델 후보군 (순서대로 시도)
    models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    
    prompt = f"""
    Extract ONE simple Korean search query (Nouns only).
    Input: {title}
    Context: {transcript[:500]}
    Rules: Remove emotional words. Return 'Person + Event'. No explanations.
    """
    
    last_error = ""
    
    # 3. 모델 순환 시도
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, safety_settings=safety_settings)
            
            # 응답 검증
            if response.text:
                return response.text.strip(), f"✨ Gemini ({model_name})"
        except Exception as e:
            last_error = str(e)
            continue # 다음 모델 시도

    # 4. 모든 모델 실패 시 -> 백업 로직 (에러 원인 포함)
    tokens = re.findall(r'[가-힣]{2,}', title)
    # 조사 제거
    cleaned = []
    for t in tokens:
        t = re.sub(r'(은|는|이|가|을|를|의)$', '', t)
        if len(t) > 1: cleaned.append(t)
        
    backup_query = " ".join(cleaned[:3]) if cleaned else title
    
    # 🚨 실패 원인을 라벨에 포함시켜서 보여줌
    return backup_query, f"🤖 Backup (Error: {last_error[:30]}...)"

# --- [5. 유틸리티 함수] ---
def normalize_korean_word(word):
    word = re.sub(r'[^가-힣0-9]', '', word)
    for j in ['은','는','이','가','을','를','의','에','에게','로','으로']:
        if word.endswith(j): return word[:-len(j)]
    return word

def extract_meaningful_tokens(text):
    raw = re.findall(r'[가-힣]{2,}', text)
    noise = ['충격','속보','긴급','오늘','지금','결국','뉴스','영상']
    return [normalize_korean_word(w) for w in raw if w not in noise]

def train_dynamic_vector_engine():
    try:
        dt = [row['video_title'] for row in supabase.table("analysis_history").select("video_title").lt("fake_prob", 40).execute().data]
        df = [row['video_title'] for row in supabase.table("analysis_history").select("video_title").gt("fake_prob", 60).execute().data]
        vector_engine.train(STATIC_TRUTH_CORPUS + dt, STATIC_FAKE_CORPUS + df)
        return len(STATIC_TRUTH_CORPUS + dt) + len(STATIC_FAKE_CORPUS + df), len(dt), len(df)
    except: 
        vector_engine.train(STATIC_TRUTH_CORPUS, STATIC_FAKE_CORPUS)
        return 0, 0, 0

def save_analysis(channel, title, prob, url, keywords):
    try: supabase.table("analysis_history").insert({"channel_name": channel, "video_title": title, "fake_prob": prob, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url": url, "keywords": keywords}).execute()
    except: pass

def render_intelligence_distribution(current_prob):
    try:
        res = supabase.table("analysis_history").select("fake_prob").execute()
        if not res.data: return
        df = pd.DataFrame(res.data)
        base = alt.Chart(df).transform_density('fake_prob', as_=['fake_prob', 'density'], extent=[0, 100], bandwidth=5).mark_area(opacity=0.3, color='#888').encode(x=alt.X('fake_prob:Q', title='가짜뉴스 확률 분포'), y=alt.Y('density:Q', title='데이터 밀도'))
        rule = alt.Chart(pd.DataFrame({'x': [current_prob]})).mark_rule(color='blue', size=3).encode(x='x')
        st.altair_chart(base + rule, use_container_width=True)
    except: pass

def colored_progress_bar(label, percent, color):
    st.markdown(f"""<div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span style="font-size: 13px; font-weight: 600; color: #555;">{label}</span><span style="font-size: 13px; font-weight: 700; color: {color};">{round(percent * 100, 1)}%</span></div><div style="background-color: #eee; border-radius: 5px; height: 8px; width: 100%;"><div style="background-color: {color}; height: 8px; width: {percent * 100}%; border-radius: 5px;"></div></div></div>""", unsafe_allow_html=True)

def render_score_breakdown(data_list):
    style = """<style>table.score-table { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; font-family: sans-serif; font-size: 14px; margin-top: 10px;} table.score-table th { background-color: #f8f9fa; color: #495057; font-weight: bold; padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; } table.score-table td { padding: 12px 15px; border-bottom: 1px solid #f0f0f0; color: #333; } table.score-table tr:last-child td { border-bottom: none; } .badge { padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-align: center; min-width: 45px; } .badge-danger { background-color: #ffebee; color: #d32f2f; } .badge-success { background-color: #e8f5e9; color: #2e7d32; } .badge-neutral { background-color: #f5f5f5; color: #757575; border: 1px solid #e0e0e0; }</style>"""
    rows = ""
    for item, score, note in data_list:
        try:
            score_num = int(score)
            badge = f'<span class="badge badge-danger">+{score_num}</span>' if score_num > 0 else f'<span class="badge badge-success">{score_num}</span>' if score_num < 0 else f'<span class="badge badge-neutral">0</span>'
        except: badge = f'<span class="badge badge-neutral">{score}</span>'
        rows += f"<tr><td>{item}<br><span style='color:#888; font-size:11px;'>{note}</span></td><td style='text-align: right;'>{badge}</td></tr>"
    st.markdown(f"{style}<table class='score-table'><thead><tr><th>분석 항목 (Silent Echo Protocol)</th><th style='text-align: right;'>변동</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def summarize_transcript(text, title):
    if not text or len(text) < 50: return "요약 불가"
    return text[:200] + "..."

def check_tag_abuse(title, hashtags, channel):
    if any(o in channel for o in OFFICIAL_CHANNELS): return 0, "공식 채널"
    if not hashtags: return 0, "태그 없음"
    return 0, "정상"

def fetch_real_transcript(info):
    try:
        url = None
        for k in ['subtitles', 'automatic_captions']:
            if k in info and 'ko' in info[k]:
                url = info[k]['ko'][0]['url']; break
        if url: return requests.get(url).text
    except: pass
    return info.get('description', '')

def fetch_comments_via_api(vid):
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        res = requests.get(url, params={'part':'snippet','videoId':vid,'key':YOUTUBE_API_KEY,'maxResults':20})
        if res.status_code == 200:
            return [i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in res.json().get('items',[])], "성공"
    except: pass
    return [], "실패"

def fetch_news_regex(query):
    try:
        rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
        raw = requests.get(rss, timeout=3).text
        items = re.findall(r'<title>(.*?)</title>', raw)
        return [{'title':t.replace("<![CDATA[","").replace("]]>","")} for t in items[1:6]]
    except: return []

def calculate_dual_match(news, query_nouns, full_text, query):
    if not news: return 0
    return 0 # Placeholder

# --- [UI Layout] ---
st.title("⚖️ Triple-Evidence Intelligence Forensic v63.5")
with st.container(border=True):
    agree = st.checkbox("동의합니다.")

url_input = st.text_input("🔗 분석할 유튜브 URL")
if st.button("🚀 정밀 분석 시작", use_container_width=True, disabled=not agree):
    if url_input: 
        with yt_dlp.YoutubeDL({'quiet':True}) as ydl:
            info = ydl.extract_info(url_input, download=False)
            title = info['title']
            transcript = fetch_real_transcript(info)
            
            # 🚨 여기가 핵심: 결과 확인
            query, source = get_gemini_search_keywords(title, transcript)
            
            # 뉴스 검색 및 로직 수행 (간소화)
            news_items = fetch_news_regex(query)
            
            st.success("분석 완료")
            st.divider()
            
            # 결과 출력
            st.info(f"🎯 **추출 검색어**: {query}")
            

[Image of magnifying glass over data]

            if "Error" in source:
                st.error(f"⚠️ **Gemini 실패 원인**: {source}")
            else:
                st.success(f"✅ **성공 출처**: {source}")
                
            st.write(f"뉴스 검색 결과: {len(news_items)}건")
            if news_items: st.dataframe(news_items)
            else: st.warning("검색된 뉴스가 없습니다.")
