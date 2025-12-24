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

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="Fact-Check Center v95.0 (Smart Cache)", layout="wide", page_icon="⚡")

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

if "debug_logs" not in st.session_state:
    st.session_state["debug_logs"] = []

# 🌟 Secrets 로드 (Streamlit Cloud 환경 기준)
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    GOOGLE_API_KEY_A = st.secrets["GOOGLE_API_KEY_A"]
    GOOGLE_API_KEY_B = st.secrets["GOOGLE_API_KEY_B"]
except:
    st.error("❌ 필수 키(API Keys)가 설정되지 않았습니다. secrets.toml 파일을 확인하세요.")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [2. 유틸리티] ---
def parse_gemini_json(text):
    """Gemini 응답에서 마크다운을 제거하고 순수 JSON 객체만 추출"""
    try:
        return json.loads(text)
    except:
        try:
            # ```json ... ``` 패턴 제거
            text = re.sub(r'```json\s*', '', text).replace('```', '')
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match: return json.loads(match.group(1))
        except: pass
    return None

def extract_video_id(url):
    """유튜브 URL에서 11자리 고유 ID 추출"""
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else None

# --- [3. 모델 탐색 & 상수] ---
@st.cache_data(ttl=3600)
def get_all_available_models(api_key):
    genai.configure(api_key=api_key)
    try:
        models = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Lite -> Flash -> Pro 순으로 가중치 부여
        models.sort(key=lambda x: 0 if 'lite' in x else 1 if 'flash' in x else 2)
        return models
    except: return ["gemini-2.0-flash", "gemini-1.5-flash"]

WEIGHT_ALGO = 0.6
WEIGHT_AI = 0.4
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', '채널A', 'TV조선', '연합뉴스', '한겨레', '조선일보', '중앙일보', '동아일보']
STATIC_TRUTH_CORPUS = ["위장전입 무혐의 판결", "임영웅 암표 강력 대응", "정희원 교수 저속노화 식단", "대전 충남 행정통합 합의"]
STATIC_FAKE_CORPUS = ["충격 폭로 경악", "긴급 속보 소름", "구속 영장 즉시 발부", "유언장 전격 공개", "사형 집행 확정"]

# --- [4. VectorEngine (내부 데이터 분석)] ---
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
        mag1 = math.sqrt(sum(a*a for a in v1))
        mag2 = math.sqrt(sum(b*b for b in v2))
        return dot/(mag1*mag2) if mag1*mag2 > 0 else 0
    def analyze_position(self, query):
        if not self.vocab: return 0, 0
        qv = self.text_to_vector(query)
        mt = max([self.cosine_similarity(qv, v) for v in self.truth_vectors] or [0])
        mf = max([self.cosine_similarity(qv, v) for v in self.fake_vectors] or [0])
        return mt, mf

vector_engine = VectorEngine()

# --- [5. Gemini Logic (Survior Mode)] ---
safety_settings_none = {HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}

def call_gemini_survivor(api_key, prompt, is_json=False):
    genai.configure(api_key=api_key)
    generation_config = {"response_mime_type": "application/json"} if is_json else {}
    all_models = get_all_available_models(api_key)
    logs = []
    for model_name in all_models:
        try:
            model = genai.GenerativeModel(model_name, generation_config=generation_config)
            response = model.generate_content(prompt, safety_settings=safety_settings_none)
            if response.text:
                logs.append(f"✅ Success: {model_name}")
                return response.text, model_name, logs
        except Exception as e:
            logs.append(f"❌ Failed ({model_name}): {str(e)[:30]}...")
            time.sleep(0.2)
            continue
    return None, "All Failed", logs

def get_gemini_search_keywords(title, transcript):
    prompt = f"Role: Fact-Check Investigator. Title: {title}. Transcript: {transcript[:10000]}. Extract ONE Korean search query for Google News (Proper Noun + Core Issue). Output: Query string only."
    res, model, logs = call_gemini_survivor(GOOGLE_API_KEY_A, prompt)
    st.session_state["debug_logs"].extend([f"[Key A] {l}" for l in logs])
    return (res.strip(), f"✨ {model}") if res else (title, "❌ Error")

def scrape_news_content_robust(url):
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5, allow_redirects=True)
        soup = BeautifulSoup(res.text, 'html.parser')
        for t in soup(['script', 'style', 'nav', 'footer', 'header']): t.decompose()
        text = " ".join([p.get_text().strip() for p in soup.find_all('p') if len(p.get_text()) > 30])
        return (text[:4000], res.url) if len(text) > 100 else (None, res.url)
    except: return None, url

def deep_verify_news(video_summary, news_url, news_snippet):
    txt, real_url = scrape_news_content_robust(news_url)
    evidence = txt if txt else news_snippet
    source = "Full Article" if txt else "Snippet"
    prompt = f"Context: {video_summary[:2000]}. News: {evidence}. Task: Score match from 0(Truth) to 100(Fake). Output JSON {{'score': int, 'reason': 'short korean reason'}}"
    res, model, logs = call_gemini_survivor(GOOGLE_API_KEY_B, prompt, is_json=True)
    st.session_state["debug_logs"].extend([f"[Key B-Verify] {l}" for l in logs])
    parsed = parse_gemini_json(res)
    if parsed: return parsed.get('score', 50), parsed.get('reason', 'N/A'), source, evidence, real_url
    return 50, "분석 실패", "Error", "", news_url

def get_gemini_verdict_final(title, transcript, news_list):
    news_summary = "\n".join([f"- {n['뉴스 제목']} (Match Score:{n['최종 점수']}, Evidence:{n['분석 근거']})" for n in news_list])
    prompt = f"Judge Final Verdict. Video: {title}. News Evidence: {news_summary}. Task: Final Fake Score (0-100). Higher = Fake. Output JSON {{'score': int, 'reason': 'korean reason'}}"
    res, model, logs = call_gemini_survivor(GOOGLE_API_KEY_B, prompt, is_json=True)
    st.session_state["debug_logs"].extend([f"[Key B-Final] {l}" for l in logs])
    parsed = parse_gemini_json(res)
    if parsed: return parsed.get('score', 50), f"{parsed.get('reason')} (By {model})"
    return 50, "판결 실패"

# --- [6. 캐싱 및 데이터베이스 관리] ---
def train_dynamic_vector_engine():
    try:
        res_t = supabase.table("analysis_history").select("video_title").lt("fake_prob", 40).execute()
        res_f = supabase.table("analysis_history").select("video_title").gt("fake_prob", 60).execute()
        dt = [row['video_title'] for row in res_t.data] if res_t.data else []
        df = [row['video_title'] for row in res_f.data] if res_f.data else []
        vector_engine.train(STATIC_TRUTH_CORPUS + dt, STATIC_FAKE_CORPUS + df)
        return len(dt)+len(df), len(dt), len(df)
    except: 
        vector_engine.train(STATIC_TRUTH_CORPUS, STATIC_FAKE_CORPUS)
        return 0, 0, 0

def check_cache(video_id):
    try:
        response = supabase.table("analysis_history").select("*").ilike("video_url", f"%{video_id}%").order("id", desc=True).limit(1).execute()
        if response.data: return response.data[0]
    except: pass
    return None

def save_analysis(channel, title, prob, url, keywords, full_report):
    try:
        data = {
            "channel_name": channel, "video_title": title, "fake_prob": prob,
            "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "video_url": url, "keywords": keywords,
            "detail_json": json.dumps(full_report, ensure_ascii=False)
        }
        supabase.table("analysis_history").insert(data).execute()
    except Exception as e:
        st.warning(f"데이터 저장 실패 (detail_json 컬럼 확인 요망): {e}")

# --- [7. UI Helper] ---
def render_final_report(final_prob, db_count, title, query, report_data, is_cached=False):
    if is_cached:
        st.success(f"🎉 **기존 분석 결과 로드 완료! (Smart Cache)**: {report_data.get('analysis_date', 'N/A')}에 이미 분석된 영상입니다.")
    
    st.subheader("🕵️ Dual-Engine Analysis Result")
    col_a, col_b, col_c = st.columns(3)
    with col_a: st.metric("최종 가짜뉴스 확률", f"{final_prob}%", delta="AI Judge Score")
    with col_b:
        icon = "🟢" if final_prob < 30 else "🔴" if final_prob > 60 else "🟠"
        verdict = "안전 (Verified)" if final_prob < 30 else "위험 (Fake/Bias)" if final_prob > 60 else "주의 (Caution)"
        st.metric("종합 AI 판정", f"{icon} {verdict}")
    with col_c: st.metric("AI Intelligence Level", f"{db_count} Nodes", delta="Active Memory")
    
    st.divider()
    st.write(f"**영상 제목:** {title}")
    st.info(f"🎯 **추출 검색 키워드:** {query}")
    
    # 점수표 렌더링
    st.write("### 📊 분석 스코어 세부 정보")
    st.table(pd.DataFrame(report_data.get('score_breakdown', []), columns=["분석 항목", "변동 점수", "상세 설명"]))
    
    # 뉴스 증거 렌더링
    st.write("### 📰 뉴스 교차 대조 증거")
    news_ev = report_data.get('news_evidence', [])
    if news_ev:
        st.dataframe(pd.DataFrame(news_ev), column_config={"원문": st.column_config.LinkColumn("기사 링크")}, hide_index=True)
    else: st.warning("대조할 기사를 찾지 못했습니다.")
        
    with st.container(border=True):
        st.write(f"⚖️ **최종 판결 요약:** {report_data.get('ai_reason', 'N/A')}")

# --- [8. 유튜브 데이터 수집기] ---
def fetch_real_transcript(info):
    try:
        subs = info.get('subtitles') or {}
        auto = info.get('automatic_captions') or {}
        merged = {**subs, **auto}
        if 'ko' in merged:
            for f in merged['ko']:
                if f['ext'] == 'vtt':
                    res = requests.get(f['url'])
                    lines = [l.strip() for l in res.text.splitlines() if l.strip() and '-->' not in l and '<' not in l]
                    return " ".join(lines[2:]), "Success"
    except: pass
    return None, "Fail"

def fetch_news_regex(query):
    try:
        rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
        raw = requests.get(rss, timeout=5).text
        items = re.findall(r'<item>(.*?)</item>', raw, re.DOTALL)
        res = []
        for i in items:
            t = re.search(r'<title>(.*?)</title>', i)
            l = re.search(r'<link>(.*?)</link>', i)
            d = re.search(r'<description>(.*?)</description>', i)
            if t and l:
                res.append({
                    'title': t.group(1).replace("<![CDATA[", "").replace("]]>", ""),
                    'desc': re.sub('<.*?>', '', d.group(1)) if d else "",
                    'link': l.group(1).strip()
                })
        return res[:5]
    except: return []

# --- [9. 메인 로직] ---
def run_forensic_main(url):
    st.session_state["debug_logs"] = []
    vid = extract_video_id(url)
    if not vid:
        st.error("유효하지 않은 유튜브 URL입니다.")
        return

    db_count, _, _ = train_dynamic_vector_engine()
    cached = check_cache(vid)
    
    if cached:
        try:
            details = json.loads(cached.get('detail_json', '{}'))
            render_final_report(cached['fake_prob'], db_count, cached['video_title'], cached.get('keywords', 'N/A'), details, is_cached=True)
            return
        except: pass

    # 신규 분석 (Progress Bar)
    my_bar = st.progress(0, text="분석 프로세스 시작 중...")
    
    with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True, 'writesubtitles': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', '제목 없음')
            uploader = info.get('uploader', '미상')
            desc = info.get('description', '')
            
            my_bar.progress(20, "1단계: 자막 및 문맥 데이터 수집 중...")
            trans, _ = fetch_real_transcript(info)
            full_text = trans if trans else desc
            
            my_bar.progress(40, "2단계: AI 수사관 가동 및 핵심 키워드 추출 중...")
            query, _ = get_gemini_search_keywords(title, full_text)
            
            my_bar.progress(60, "3단계: 뉴스 딥 웹 크롤링 및 팩트체크 중...")
            news_items = fetch_news_regex(query)
            news_ev = []; max_match = 0
            for item in news_items[:3]:
                score, reason, src, _, real_url = deep_verify_news(full_text, item['link'], item['desc'])
                if score > max_match: max_match = score
                news_ev.append({"뉴스 제목": item['title'], "일치도": f"{score}%", "최종 점수": score, "분석 근거": reason, "원문": real_url})
            
            news_penalty = -30 if max_match <= 20 else (30 if max_match >= 80 else 0)
            
            ts, fs = vector_engine.analyze_position(query + " " + title)
            t_impact = int(ts * 30) * -1; f_impact = int(fs * 30)
            
            my_bar.progress(90, "4단계: AI 판사 최종 판결문 작성 중...")
            ai_score, ai_reason = get_gemini_verdict_final(title, full_text, news_ev)
            
            algo_base = 50 + t_impact + f_impact + news_penalty
            final_prob = max(1, min(99, int(algo_base * WEIGHT_ALGO + ai_score * WEIGHT_AI)))
            
            full_report = {
                "score_breakdown": [
                    ["기본 중립 점수", 50, "중립 상태에서 분석 시작"],
                    ["진실 데이터 유사성", t_impact, "내부 DB 진실 데이터와 일치도"],
                    ["거짓 패턴 유사성", f_impact, "내부 DB 가짜 데이터와 일치도"],
                    ["뉴스 교차 대조 결과", news_penalty, "뉴스 보도 내용과의 부합 여부"],
                    ["AI 최종 추론", ai_score, ai_reason]
                ],
                "news_evidence": news_ev,
                "ai_reason": ai_reason,
                "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            save_analysis(uploader, title, final_prob, url, query, full_report)
            my_bar.empty()
            render_final_report(final_prob, db_count, title, query, full_report, is_cached=False)

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# --- [10. UI 레이아웃] ---
st.title("⚖️ Fact-Check Center v95.0")
st.markdown("> **유튜브 URL 하나로 진실을 가려내는 AI 에이전트**")

with st.container(border=True):
    st.caption("🛡️ 본 서비스는 AI 및 알고리즘 기반 보조 도구로, 법적 효력이 없음을 고지합니다.")
    url_input = st.text_input("🔗 분석할 유튜브 URL을 입력하세요", placeholder="https://www.youtube.com/watch?v=...")
    analyze_btn = st.button("🚀 정밀 분석 시작 (무료 AI 쿼터 사용)", use_container_width=True)

if analyze_btn and url_input:
    run_forensic_main(url_input)

st.divider()
st.subheader("🗂️ 최근 분석 히스토리")
try:
    history = supabase.table("analysis_history").select("analysis_date, video_title, fake_prob, keywords").order("id", desc=True).limit(10).execute()
    if history.data:
        st.dataframe(pd.DataFrame(history.data), hide_index=True, use_container_width=True)
    else: st.info("아직 분석된 데이터가 없습니다.")
except: pass

# Admin Section
with st.expander("🔐 관리자 접속"):
    pwd = st.text_input("Admin PW", type="password")
    if pwd == ADMIN_PASSWORD:
        st.session_state["is_admin"] = True
        st.success("디버그 모드 활성화")
        if st.session_state["debug_logs"]:
            st.text_area("System Logs", "\n".join(st.session_state["debug_logs"]), height=200)
