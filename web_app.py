import streamlit as st
from supabase import create_client, Client
import re
import requests
import time
import random
import math
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from datetime import datetime
from collections import Counter
import yt_dlp
import pandas as pd
import altair as alt
import json
from bs4 import BeautifulSoup

# --- [1. System Setup] ---
st.set_page_config(page_title="Fact-Check Center v92.1 (Hotfix)", layout="wide", page_icon="🩹")

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

# 🌟 Secrets Load
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    GOOGLE_API_KEY_A = st.secrets["GOOGLE_API_KEY_A"]
    GOOGLE_API_KEY_B = st.secrets["GOOGLE_API_KEY_B"]
except:
    st.error("❌ API Keys are missing in secrets.toml")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [2. Utilities] ---
def parse_gemini_json(text):
    try: return json.loads(text)
    except:
        try:
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match: return json.loads(match.group(1))
        except: pass
    return None

# --- [3. Model Pre-check] ---
CANDIDATE_MODELS = ["gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

def find_best_model(api_key):
    genai.configure(api_key=api_key)
    safety = {HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE}
    for model_name in CANDIDATE_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("hi", safety_settings=safety)
            if response.text: return model_name
        except: continue
    return "gemini-2.0-flash"

if "best_model_name" not in st.session_state:
    with st.spinner("🚀 Connecting to AI Model..."):
        st.session_state["best_model_name"] = find_best_model(GOOGLE_API_KEY_A)

# --- [4. Constants] ---
WEIGHT_ALGO = 0.6
WEIGHT_AI = 0.4
VITAL_KEYWORDS = ['위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '내란', '간첩']
CRITICAL_STATE_KEYWORDS = ['별거', '이혼', '파경', '사망', '위독', '구속', '체포', '실형', '불화', '폭로', '충격', '논란', '중태', '심정지', '뇌사', '압수수색', '소환', '파산', '빚더미', '전과', '감옥', '간첩']
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']
STATIC_TRUTH_CORPUS = ["박나래 위장전입 무혐의", "임영웅 암표 대응", "정희원 저속노화", "대전 충남 통합", "선거 출마 선언"]
STATIC_FAKE_CORPUS = ["충격 폭로 경악", "긴급 속보 소름", "충격 발언 논란", "구속 영장 발부", "영상 유출", "계시 예언", "사형 집행", "위독설"]

# --- [5. VectorEngine] ---
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
    def text_to_vector(self, text, vocabulary=None):
        target_vocab = vocabulary if vocabulary else self.vocab
        c = Counter(self.tokenize(text))
        return [c[w] for w in target_vocab]
    def cosine_similarity(self, v1, v2):
        dot = sum(a*b for a,b in zip(v1,v2))
        mag = math.sqrt(sum(a*a for a in v1)) * math.sqrt(sum(b*b for b in v2))
        return dot/mag if mag>0 else 0
    def analyze_position(self, query):
        qv = self.text_to_vector(query)
        mt = max([self.cosine_similarity(qv, v) for v in self.truth_vectors] or [0])
        mf = max([self.cosine_similarity(qv, v) for v in self.fake_vectors] or [0])
        return mt, mf
    def compute_content_similarity(self, text1, text2):
        tokens1 = self.tokenize(text1); tokens2 = self.tokenize(text2)
        local_vocab = sorted(list(set(tokens1 + tokens2)))
        if not local_vocab: return 0.0
        v1 = self.text_to_vector(text1, local_vocab)
        v2 = self.text_to_vector(text2, local_vocab)
        return self.cosine_similarity(v1, v2)

vector_engine = VectorEngine()

# --- [6. Gemini Logic] ---
safety_settings_none = {HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}

def call_gemini_fast(api_key, prompt, is_json=False):
    genai.configure(api_key=api_key)
    target_model = st.session_state.get("best_model_name", "gemini-2.0-flash")
    generation_config = {"response_mime_type": "application/json"} if is_json else {}
    try:
        model = genai.GenerativeModel(target_model, generation_config=generation_config)
        response = model.generate_content(prompt, safety_settings=safety_settings_none)
        return response.text, target_model
    except Exception as e:
        try:
            fallback = "gemini-2.0-flash"
            model = genai.GenerativeModel(fallback, generation_config=generation_config)
            response = model.generate_content(prompt, safety_settings=safety_settings_none)
            return response.text, fallback
        except: return None, str(e)

# [Engine A]
def get_gemini_search_keywords(title, transcript):
    prompt = f"Role: Investigator. Input: {title}, {transcript[:10000]}. Task: Extract ONE Korean Google News search query (Noun+Issue). Output: Query string only."
    result_text, model = call_gemini_fast(GOOGLE_API_KEY_A, prompt)
    return (result_text.strip(), f"✨ {model}") if result_text else (title, "❌ Error")

# [Web Crawler]
def scrape_news_content_robust(google_url):
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        response = session.get(google_url, timeout=5, allow_redirects=True)
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer']): tag.decompose()
        text = " ".join([p.get_text().strip() for p in soup.find_all('p') if len(p.get_text()) > 30])
        return (text[:4000], response.url) if len(text) > 50 else (None, response.url)
    except: return None, google_url

# [Engine B]
def deep_verify_news(video_summary, news_url, news_snippet):
    scraped_text, real_url = scrape_news_content_robust(news_url)
    evidence = scraped_text if scraped_text else news_snippet
    source = "Full Article" if scraped_text else "Snippet"
    
    prompt = f"Compare: {video_summary[:2000]} vs {evidence}. Task: Check if news confirms video claim. Output JSON {{score:int, reason:korean_str}}"
    result_text, model = call_gemini_fast(GOOGLE_API_KEY_B, prompt, is_json=True)
    res = parse_gemini_json(result_text)
    
    if res: return res.get('score', 0), res.get('reason', 'N/A'), source, evidence, real_url
    return 0, "Analysis Error", "Error", "", news_url

# [Engine B Final]
def get_gemini_verdict_final(title, transcript, news_list):
    summary = "\n".join([f"- {n['뉴스 제목']} (Score:{n['최종 점수']})" for n in news_list])
    prompt = f"Role: Judge. Video: {title}. Evidence: {summary}. Task: Verify truth. Output JSON {{score:int, reason:korean_str}}"
    result_text, model = call_gemini_fast(GOOGLE_API_KEY_B, prompt, is_json=True)
    res = parse_gemini_json(result_text)
    if res: return res.get('score', 50), f"{res.get('reason')} ({model})"
    return 50, "Judge Error"

# --- [7. Utilities] ---
def normalize_korean_word(word):
    word = re.sub(r'[^가-힣0-9]', '', word)
    for j in ['은','는','이','가','을','를','의','에','에게','로','으로']:
        if word.endswith(j): return word[:-len(j)]
    return word

def extract_meaningful_tokens(text):
    return [normalize_korean_word(w) for w in re.findall(r'[가-힣]{2,}', text) if w not in ['충격','속보','뉴스']]

def train_dynamic_vector_engine():
    try:
        res_t = supabase.table("analysis_history").select("video_title").lt("fake_prob", 40).execute()
        res_f = supabase.table("analysis_history").select("video_title").gt("fake_prob", 60).execute()
        dt = [row['video_title'] for row in res_t.data] if res_t.data else []
        df = [row['video_title'] for row in res_f.data] if res_f.data else []
        vector_engine.train(STATIC_TRUTH_CORPUS + dt, STATIC_FAKE_CORPUS + df)
        return len(dt)+len(df), dt, df
    except: 
        vector_engine.train(STATIC_TRUTH_CORPUS, STATIC_FAKE_CORPUS)
        return 0, [], []

def check_db_similarity(query, truth, fake):
    return vector_engine.analyze_position(query)

def save_analysis(channel, title, prob, url, keywords):
    try: supabase.table("analysis_history").insert({"channel_name": channel, "video_title": title, "fake_prob": prob, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url": url, "keywords": keywords}).execute()
    except: pass

def render_intelligence_distribution(current_prob):
    try:
        res = supabase.table("analysis_history").select("fake_prob").execute()
        if res.data:
            df = pd.DataFrame(res.data)
            c = alt.Chart(df).transform_density('fake_prob', as_=['fake_prob', 'density'], extent=[0, 100]).mark_area(opacity=0.3).encode(x='fake_prob:Q', y='density:Q')
            st.altair_chart(c, use_container_width=True)
    except: pass

def colored_progress_bar(label, percent, color):
    st.markdown(f"**{label}**: {int(percent*100)}%", unsafe_allow_html=True)
    st.progress(min(percent, 1.0))

def render_score_breakdown(data):
    df = pd.DataFrame(data, columns=["항목", "점수", "설명"])
    st.table(df)

def summarize_transcript(text, title, max_sentences=3):
    return text[:500] + "..." 

def clean_html_regex(text):
    return re.sub('<.*?>', '', text).strip() if text else ""

def detect_ai_content(info):
    return False, "미감지"

def check_is_official(name):
    return any(o in name.upper().replace(" ","") for o in OFFICIAL_CHANNELS)

def count_sensational_words(text):
    return sum(text.count(w) for w in ['충격','경악','속보'])

def check_tag_abuse(title, tags, uploader):
    if not tags: return 0, "없음"
    return 0, "양호"

# [Hotfix: Safe Transcript Fetch]
def fetch_real_transcript(info_dict):
    try:
        url = None
        # Handle cases where subtitles might be None
        subs = info_dict.get('subtitles') or {}
        auto_subs = info_dict.get('automatic_captions') or {}
        
        # Merge dictionaries safely
        merged = {**subs, **auto_subs}
        
        if 'ko' in merged:
            for fmt in merged['ko']:
                if fmt['ext'] == 'vtt': url = fmt['url']; break
                
        if url:
            res = requests.get(url)
            if res.status_code == 200:
                lines = [l.strip() for l in res.text.splitlines() if l.strip() and '-->' not in l and '<' not in l]
                return " ".join(lines[2:]), "Success"
    except: pass
    return None, "Fail"

# [Hotfix: Safe Comment Fetch]
def fetch_comments_via_api(video_id):
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        res = requests.get(url, params={'part': 'snippet', 'videoId': video_id, 'key': YOUTUBE_API_KEY, 'maxResults': 20})
        if res.status_code == 200:
            data = res.json()
            items = []
            for i in data.get('items', []):
                # Ensure snippet path exists
                snip = i.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
                if snip and 'textDisplay' in snip:
                    items.append(snip['textDisplay'])
            return items, "Success"
    except: pass
    return [], "Fail"

def fetch_news_regex(query):
    news_res = []
    try:
        rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
        raw = requests.get(rss, timeout=5).text
        items = re.findall(r'<item>(.*?)</item>', raw, re.DOTALL)
        for item in items[:10]:
            t = re.search(r'<title>(.*?)</title>', item)
            d = re.search(r'<description>(.*?)</description>', item)
            l = re.search(r'<link>(.*?)</link>', item)
            if t and l:
                nt = t.group(1).replace("<![CDATA[", "").replace("]]>", "")
                nl = l.group(1).strip()
                nd = clean_html_regex(d.group(1)) if d else ""
                news_res.append({'title': nt, 'desc': nd, 'link': nl})
    except: pass
    return news_res

def run_forensic_main(url):
    my_bar = st.progress(0, text="분석 시작...")
    
    db_count, db_truth, db_fake = train_dynamic_vector_engine()
    
    vid = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if not vid:
        st.error("유효하지 않은 유튜브 URL입니다.")
        return
    vid = vid.group(1)

    with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            if not info: raise Exception("Video info failed")
            
            title = info.get('title', '제목 없음')
            uploader = info.get('uploader', '미상')
            tags = info.get('tags', [])
            desc = info.get('description', '')
            
            my_bar.progress(20, "자막 수집 중...")
            trans, t_status = fetch_real_transcript(info)
            full_text = trans if trans else desc
            
            my_bar.progress(40, "키워드 분석 중...")
            query, source = get_gemini_search_keywords(title, full_text)
            
            # DB & Algo
            ts, fs = check_db_similarity(query+" "+title, STATIC_TRUTH_CORPUS + db_truth, STATIC_FAKE_CORPUS + db_fake)
            t_impact = int(ts*30)*-1; f_impact = int(fs*30)
            
            my_bar.progress(60, "뉴스 크롤링 중...")
            news_items = fetch_news_regex(query)
            news_ev = []; max_match = 0
            
            for idx, item in enumerate(news_items[:3]):
                ai_s, ai_r, src, txt, real_url = deep_verify_news(full_text, item['link'], item['desc'])
                if ai_s > max_match: max_match = ai_s
                news_ev.append({
                    "뉴스 제목": item['title'],
                    "일치도": f"{ai_s}%",
                    "최종 점수": f"{ai_s}%",
                    "분석 근거": ai_r,
                    "원문": real_url
                })
            
            news_score = -30 if max_match >= 70 else (-10 if max_match >= 50 else 10) if news_ev else 0
            
            my_bar.progress(80, "여론 분석 중...")
            cmts, c_s = fetch_comments_via_api(vid)
            sent_score = 0
            
            my_bar.progress(90, "최종 판결 중...")
            ai_score, ai_reason = get_gemini_verdict_final(title, full_text, news_ev)
            
            if t_impact == 0 and f_impact == 0 and not news_ev:
                ai_score = int((ai_score + 50) / 2)
            
            base_score = 50 + t_impact + f_impact + news_score + sent_score
            final_prob = max(1, min(99, int(base_score * WEIGHT_ALGO + ai_score * WEIGHT_AI)))
            
            save_analysis(uploader, title, final_prob, url, query)
            my_bar.empty()
            
            # [UI Output]
            st.subheader("🕵️ 분석 결과")
            col1, col2, col3 = st.columns(3)
            col1.metric("가짜뉴스 확률", f"{final_prob}%")
            col2.metric("AI 판정", "위험" if final_prob > 60 else "안전" if final_prob < 40 else "주의")
            col3.metric("DB 누적", f"{db_count} 건")
            
            render_intelligence_distribution(final_prob)
            
            st.write("---")
            st.write(f"**영상 제목:** {title}")
            st.info(f"**추출 키워드:** {query}")
            
            st.write("### 📊 상세 점수표")
            render_score_breakdown([
                ["기본 점수", 50, "중립 시작"],
                ["DB 유사도(진실)", t_impact, "내부 데이터 매칭"],
                ["DB 유사도(거짓)", f_impact, "내부 데이터 매칭"],
                ["뉴스 검증", news_score, "Deep-Web 크롤링 결과"],
                ["AI 최종 판결", ai_score, ai_reason]
            ])
            
            st.write("### 📰 뉴스 교차 검증")
            if news_ev:
                st.dataframe(pd.DataFrame(news_ev), column_config={"원문": st.column_config.LinkColumn("링크")}, hide_index=True)
            else:
                st.warning("관련 뉴스를 찾을 수 없습니다.")

        except Exception as e:
            st.error(f"오류 발생: {e}")

# --- [UI] ---
st.title("⚖️ Fact-Check Center v92.1 (Hotfix)")
url = st.text_input("YouTube URL")
if st.button("분석 시작") and url: run_forensic_main(url)

# Admin
with st.expander("Admin"):
    if st.text_input("PW", type="password") == ADMIN_PASSWORD:
        st.session_state["is_admin"] = True
        st.success("Admin Logged In")
