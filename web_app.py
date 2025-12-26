import streamlit as st
from supabase import create_client, Client
import re
import requests
import time
import random
import math
import os
import json
from collections import Counter
from datetime import datetime

# --- [라이브러리 임포트] ---
from mistralai import Mistral
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import yt_dlp
import pandas as pd
import altair as alt
from bs4 import BeautifulSoup

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="유튜브 가짜뉴스 판독기 (Triple Engine)", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
        .block-container { padding-top: 3.5rem !important; padding-bottom: 5rem; }
        .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #eee; text-align: center; }
        div[data-testid="stMetricValue"] { font-size: 1.3rem !important; }
        h1 { font-size: 1.8rem !important; padding-bottom: 10px; }
        h3 { font-size: 1.2rem !important; margin-top: 20px !important; }
        .risk-badge { padding: 5px 10px; border-radius: 5px; font-weight: bold; color: white; }
    </style>
""", unsafe_allow_html=True)

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False
if "debug_logs" not in st.session_state:
    st.session_state["debug_logs"] = []

# 🌟 Secrets 로드
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    GOOGLE_API_KEY_A = st.secrets["GOOGLE_API_KEY_A"]
    GOOGLE_API_KEY_B = st.secrets["GOOGLE_API_KEY_B"]
except:
    st.error("❌ secrets.toml 파일에 API Key 설정이 필요합니다.")
    st.stop()

@st.cache_resource
def init_clients():
    su = create_client(SUPABASE_URL, SUPABASE_KEY)
    mi = Mistral(api_key=MISTRAL_API_KEY)
    return su, mi

supabase, mistral_client = init_clients()

# --- [2. 모델 정의] ---
MISTRAL_MODELS = ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"]

def get_gemini_models_dynamic(api_key):
    genai.configure(api_key=api_key)
    try:
        models = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        models.sort(key=lambda x: 0 if 'flash' in x else 1 if 'pro' in x else 2)
        return models
    except: return ["gemini-2.0-flash", "gemini-1.5-flash"]

# --- [핵심 기술: 벡터 엔진] ---
class VectorEngine:
    def __init__(self):
        self.truth_vectors = []
        self.fake_vectors = []
        self.model_name = "models/text-embedding-004" 

    def get_embedding(self, text):
        try:
            genai.configure(api_key=GOOGLE_API_KEY_A)
            result = genai.embed_content(
                model=self.model_name,
                content=text[:2000],
                task_type="retrieval_document"
            )
            return result['embedding']
        except:
            return [0] * 768

    def load_pretrained_vectors(self, truth_vecs, fake_vecs):
        self.truth_vectors = truth_vecs
        self.fake_vectors = fake_vecs

    def train_static(self, truth_text, fake_text):
        self.truth_vectors.extend([self.get_embedding(t) for t in truth_text])
        self.fake_vectors.extend([self.get_embedding(t) for t in fake_text])

    def cosine_similarity(self, v1, v2):
        if not v1 or not v2: return 0
        dot = sum(a*b for a,b in zip(v1,v2))
        mag1 = math.sqrt(sum(a*a for a in v1))
        mag2 = math.sqrt(sum(b*b for b in v2))
        if mag1 == 0 or mag2 == 0: return 0
        return dot / (mag1 * mag2)

    def analyze(self, query):
        query_vec = self.get_embedding(query)
        def calibrate(score):
            baseline = 0.75 
            if score < baseline: return 0.0
            return (score - baseline) / (1.0 - baseline)
        raw_t = max([self.cosine_similarity(query_vec, v) for v in self.truth_vectors] or [0])
        raw_f = max([self.cosine_similarity(query_vec, v) for v in self.fake_vectors] or [0])
        return calibrate(raw_t), calibrate(raw_f)

vector_engine = VectorEngine()

# --- [3. 유틸리티] ---
def parse_llm_json(text):
    try: parsed = json.loads(text)
    except:
        try:
            text = re.sub(r'```json\s*', '', text); text = re.sub(r'```', '', text)
            match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if match: parsed = json.loads(match.group(1))
            else: return None
        except: return None
    if isinstance(parsed, list): return parsed[0] if len(parsed) > 0 and isinstance(parsed[0], dict) else None
    if isinstance(parsed, dict): return parsed
    return None

def determine_risk_level(prob):
    if prob >= 70: return "⛔ 위험 (High Risk)", "#d32f2f"
    elif prob >= 40: return "⚠️ 주의 (Caution)", "#f57c00"
    return "✅ 안전 (Safe)", "#388e3c"

def colored_bar_html(label, score, color):
    pct = min(100, max(0, int(score * 100)))
    return f"""<div style="margin-bottom: 6px;"><div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: #444;"><span>{label}</span><span>{pct}%</span></div><div style="width: 100%; background-color: #e0e0e0; border-radius: 6px; height: 8px; margin-top: 2px;"><div style="width: {pct}%; background-color: {color}; height: 8px; border-radius: 6px;"></div></div></div>"""

# --- [4. AI Logic] ---
def call_triple_survivor(prompt, is_json=False):
    logs = []
    response_format = {"type": "json_object"} if is_json else None
    
    # Mistral
    for model_name in MISTRAL_MODELS:
        try:
            resp = mistral_client.chat.complete(
                model=model_name, messages=[{"role": "user", "content": prompt}],
                response_format=response_format, temperature=0.2
            )
            if resp.choices:
                logs.append(f"✅ Success (Mistral): {model_name}")
                return resp.choices[0].message.content, model_name, logs
        except Exception as e:
            logs.append(f"❌ Mistral Failed: {str(e)[:20]}")
            continue

    # Gemini Fallback
    generation_config = {"response_mime_type": "application/json"} if is_json else {}
    for key_name, key_val in [("Key A", GOOGLE_API_KEY_A), ("Key B", GOOGLE_API_KEY_B)]:
        logs.append(f"⚠️ Mistral Failed -> Gemini {key_name} 투입")
        genai.configure(api_key=key_val)
        models = get_gemini_models_dynamic(key_val)
        for model_name in models:
            try:
                model = genai.GenerativeModel(model_name, generation_config=generation_config)
                resp = model.generate_content(prompt)
                if resp.text:
                    logs.append(f"✅ Success (Gemini {key_name}): {model_name}")
                    return resp.text, f"{model_name} ({key_name})", logs
            except: continue
    return None, "All Failed", logs

# --- [5. Data Constants] ---
WEIGHT_ALGO = 0.85
WEIGHT_AI = 0.15
OFFICIAL_CHANNELS = ['MBC','KBS','SBS','EBS','YTN','JTBC','TVCHOSUN','MBN','CHANNEL A','연합뉴스','YONHAP','한겨레','경향','조선','중앙','동아']
CRITICAL_STATE_KEYWORDS = ['별거','이혼','파경','사망','위독','구속','체포','실형','불화','폭로','충격','논란','심정지','뇌사','압수수색','감옥']
STATIC_TRUTH = ["박나래 위장전입 무혐의", "임영웅 암표 대응", "정희원 저속노화", "선거 출마 선언"]
STATIC_FAKE = ["충격 폭로 경악", "긴급 속보 소름", "구속 영장 발부", "사형 집행", "위독설"]

def get_keywords(title, trans):
    prompt = f"""You are a Fact-Check Investigator. [Input] {title}, {trans[:10000]}. [Task] Generate 3 diverse Google News queries (Specific, Contextual, Keywords). [Output JSON] {{ "queries": ["query1", "query2", "query3"] }}"""
    res, model, logs = call_triple_survivor(prompt, is_json=True)
    st.session_state["debug_logs"].extend([f"[Key] {l}" for l in logs])
    parsed = parse_llm_json(res)
    if parsed and 'queries' in parsed and isinstance(parsed['queries'], list): return parsed['queries'], model
    return [title, title+" 팩트체크", title+" 뉴스"], model

def scrape_news(url):
    try:
        res = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'html.parser')
        for t in soup(['script','style','nav','footer']): t.decompose()
        text = " ".join([p.get_text().strip() for p in soup.find_all('p') if len(p.get_text())>30])
        return (text[:3000], res.url) if len(text)>100 else (None, res.url)
    except: return None, url

def verify_news(summary, link, snippet):
    txt, real_url = scrape_news(link)
    ev = txt if txt else snippet
    prompt = f"Compare Video({summary[:1000]}) vs News({ev}). Match(90-100)/Related(40-60)/Mismatch(0-10). Output JSON: {{ \"score\": int, \"reason\": \"korean short\" }}"
    res, _, logs = call_triple_survivor(prompt, is_json=True)
    st.session_state["debug_logs"].extend([f"[Verify] {l}" for l in logs])
    p = parse_llm_json(res)
    return (p['score'], p['reason'], "Full" if txt else "Snippet", real_url) if p else (0, "Err", "Err", link)

def judge_final(title, trans, evidences):
    ev_text = "".join([f"- {e['뉴스 제목']} (Score:{e['최종 점수']}, Reason:{e['분석 근거']})\n" for e in evidences])
    prompt = f"Judge Video: {title}. Evidence: {ev_text}. Decide Fake Probability (0-100). Output JSON: {{ \"score\": int, \"reason\": \"korean explanation\" }}"
    res, model, logs = call_triple_survivor(prompt, is_json=True)
    st.session_state["debug_logs"].extend([f"[Judge] {l}" for l in logs])
    p = parse_llm_json(res)
    return (p['score'], f"{p['reason']} ({model})") if p else (50, "Failed")

def generate_comprehensive_summary(title, final_prob, news_ev, red_cnt, ai_reason, risk_text):
    prompt = f"""Fact-Check AI Analyst. Video: {title}. Prob: {final_prob}% ({risk_text}). News Evidence: {len(news_ev)}. Red Comments: {red_cnt}. AI Reason: {ai_reason}.
    Task: Write a final summary report in Korean. (Why score, Fact check result, Advice)."""
    res, _, _ = call_triple_survivor(prompt, is_json=False)
    return res if res else "종합 분석 실패"

# --- [6. Helper Functions] ---
def normalize(w): return re.sub(r'은$|는$|이$|가$|을$|를$|의$|에$|로$', '', re.sub(r'[^가-힣0-9]', '', w))
def get_tokens(t): return [normalize(w) for w in re.findall(r'[가-힣]{2,}', t) if w not in ['충격','속보','뉴스']]
def get_top_kw(t): return Counter(get_tokens(t)).most_common(5)
def check_official(n): return any(o in n.upper().replace(" ","") for o in OFFICIAL_CHANNELS)
def count_agitation(t): return sum(t.count(w) for w in ['충격','경악','실체','폭로','속보','소름'])
def check_red_flags(cmts): 
    d = [k for c in cmts for k in ['가짜','주작','구라','허위','선동'] if k in c]
    return len(d), list(set(d))

# --- [Data Fetching & DB] ---
def fetch_transcript(info):
    try:
        url = None
        for fmt in (info.get('subtitles') or {}).get('ko', []) + (info.get('automatic_captions') or {}).get('ko', []):
            if fmt['ext'] == 'vtt': url = fmt['url']; break
        if url: return " ".join([l.strip() for l in requests.get(url).text.splitlines() if l.strip() and '-->' not in l and '<' not in l]), "Success"
    except: pass
    return None, "Fail"

def fetch_comments(vid):
    try:
        res = requests.get("https://www.googleapis.com/youtube/v3/commentThreads", params={'part':'snippet','videoId':vid,'key':YOUTUBE_API_KEY,'maxResults':50})
        if res.status_code==200: return [str(i['snippet']['topLevelComment']['snippet']['textDisplay']) for i in res.json().get('items',[])]
    except: pass
    return []

def fetch_news(q):
    try:
        raw = requests.get(f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=ko&gl=KR", timeout=5).text
        items = re.findall(r'<item>(.*?)</item>', raw, re.DOTALL)
        res = []
        for i in items[:10]:
            t = re.search(r'<title>(.*?)</title>', i); l = re.search(r'<link>(.*?)</link>', i)
            if t and l: res.append({'title':t.group(1).replace("<![CDATA[","").replace("]]>",""), 'link':l.group(1).strip()})
        return res
    except: return []

def analyze_comments(cmts, ctx):
    if not cmts: return [], 0, "데이터 부족"
    safe_cmts = " ".join([str(c) for c in cmts])
    top = Counter(get_tokens(safe_cmts)).most_common(5)
    ctx_set = set(get_tokens(ctx))
    score = int(sum(1 for w,c in top if w in ctx_set)/len(top)*100) if top else 0
    return [f"{w}({c})" for w,c in top], score, "높음" if score>=60 else "보통" if score>=20 else "낮음"

# [수정] 테이블 이름을 'analysis_archive_v2'로 변경했습니다.
@st.cache_data(ttl=3600)
def fetch_db_vectors():
    try:
        # v2 테이블 조회
        res = supabase.table("analysis_archive_v2").select("video_title, fake_prob, vector_json").execute()
        if not res.data: return [], [], 0
        dt_vecs, df_vecs = [], []
        for row in res.data:
            if row.get('vector_json'):
                vec = json.loads(row['vector_json']) if isinstance(row['vector_json'], str) else row['vector_json']
                if row['fake_prob'] < 40: dt_vecs.append(vec)
                elif row['fake_prob'] > 60: df_vecs.append(vec)
        return dt_vecs, df_vecs, len(res.data)
    except: return [], [], 0

def train_engine_wrapper():
    dt_vecs, df_vecs, count = fetch_db_vectors()
    vector_engine.load_pretrained_vectors(dt_vecs, df_vecs)
    vector_engine.train_static(STATIC_TRUTH, STATIC_FAKE)
    return count, [], []

# [수정] 테이블 이름을 'analysis_archive_v2'로 변경했습니다.
def save_db(ch, ti, pr, url, kw, detail):
    try: 
        embedding = vector_engine.get_embedding(kw + " " + ti)
        data_to_insert = {
            "channel_name":ch, "video_title":ti, "fake_prob":pr, "video_url":url, 
            "keywords":kw, "detail_json":detail, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "vector_json": embedding
        }
        # v2 테이블에 저장
        supabase.table("analysis_archive_v2").insert(data_to_insert).execute()
        st.toast("✅ DB 저장 완료!", icon="💾")
    except Exception as e: 
        st.error(f"❌ 데이터베이스 저장 실패: {e}")
        print(f"DB Error: {e}")

# --- [UI Components] ---
def render_score_breakdown(data_list):
    style = """<style>table.score-table { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; font-family: sans-serif; font-size: 14px; margin-top: 10px;} table.score-table th { background-color: #f8f9fa; color: #495057; font-weight: bold; padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; } table.score-table td { padding: 12px 15px; border-bottom: 1px solid #f0f0f0; color: #333; } table.score-table tr:last-child td { border-bottom: none; } .badge { padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-align: center; min-width: 45px; } .badge-danger { background-color: #ffebee; color: #d32f2f; } .badge-success { background-color: #e8f5e9; color: #2e7d32; } .badge-neutral { background-color: #f5f5f5; color: #757575; border: 1px solid #e0e0e0; }</style>"""
    rows = ""
    for item, score, note in data_list:
        try:
            score_num = int(score)
            badge = f'<span class="badge badge-danger">+{score_num}</span>' if score_num > 0 else f'<span class="badge badge-success">{score_num}</span>' if score_num < 0 else f'<span class="badge badge-neutral">0</span>'
        except: badge = f'<span class="badge badge-neutral">{score}</span>'
        rows += f"<tr><td>{item}<br><span style='color:#888; font-size:11px;'>{note}</span></td><td style='text-align: right;'>{badge}</td></tr>"
    st.markdown(f"{style}<table class='score-table'><thead><tr><th>분석 항목</th><th style='text-align: right;'>변동</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def render_intelligence_distribution(current_prob):
    try:
        res = supabase.table("analysis_history").select("fake_prob").execute()
        if not res.data: return
        df = pd.DataFrame(res.data)
        base = alt.Chart(df).transform_density('fake_prob', as_=['fake_prob', 'density'], extent=[0, 100], bandwidth=5).mark_area(opacity=0.3, color='#888').encode(x=alt.X('fake_prob:Q', title='가짜뉴스 확률 분포'), y=alt.Y('density:Q', title='밀도'))
        rule = alt.Chart(pd.DataFrame({'x': [current_prob]})).mark_rule(color='red', size=3).encode(x='x')
        st.altair_chart(base + rule, use_container_width=True)
    except: pass

def render_report_full_ui(prob, db_count, title, channel, data, is_cached=False):
    st.divider()
    if is_cached: st.info(f"💾 과거 분석 기록 호출됨 (총 DB 데이터: {db_count}개)")
    
    risk_text, risk_color = determine_risk_level(prob)
    
    # 1. Hero
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1: st.metric("🔥 가짜뉴스 확률", f"{prob}%")
    with c2: st.markdown(f"<div style='text-align:center; padding:10px; border-radius:10px; background-color:{risk_color}; color:white; font-weight:bold; font-size:1.1rem; margin-top:5px;'>{risk_text}</div>", unsafe_allow_html=True)
    with c3: st.metric("🗄️ 누적 DB", f"{db_count}건")
    
    # 2. Conclusion First
    st.subheader("📝 AI 최종 종합 리포트")
    with st.container(border=True):
        st.markdown(f"**📢 AI Analyst Comment:**\n\n{data.get('final_summary', '데이터 없음')}")

    # 3. Video Info
    st.subheader("ℹ️ 영상 기본 정보")
    with st.container(border=True):
        st.write(f"**📺 {title}**")
        st.caption(f"채널: {channel} | 분석일: {datetime.now().strftime('%Y-%m-%d')}")
        st.info(f"**요약:** {data.get('summary', '요약 없음')}")
        with st.expander("상세 메타데이터"):
            st.dataframe(pd.DataFrame([data.get('meta', {})]), use_container_width=True, hide_index=True)

    # 4. Evidence Tabs
    st.subheader("🔍 상세 증거 및 분석 데이터")
    tab_news, tab_data, tab_ai = st.tabs(["📰 뉴스 팩트체크", "📊 데이터/여론", "🤖 AI 기술적 판단"])
    
    with tab_news:
        st.markdown("###### 🗝️ AI 검색 키워드 (3-Way Strategy)")
        if data.get('query_list'):
            st.caption("AI가 추출한 3가지 전략 키워드: " + " | ".join([f"`{q}`" for q in data['query_list']]))
        if data.get('query'):
            st.success(f"✅ 뉴스 매칭 성공 키워드: **{data['query']}**")
        
        st.divider()
        st.write("###### [증거 2] 뉴스 대조 결과 (Top 5)")
        if data.get('news_evidence'):
            for news in data['news_evidence']:
                with st.expander(f"{news['일치도']} {news['뉴스 제목']}"):
                    st.write(f"**🕵️ 분석:** {news['분석 근거']}")
                    st.caption(f"출처: {news['비고']}")
                    st.link_button("🔗 원문 보기", news['원문'])
        else: st.warning("관련 뉴스 없음")

    with tab_data:
        st.write("###### [증거 1] 데이터 유사도 분석")
        c1, c2 = st.columns(2)
        with c1: st.markdown(colored_bar_html("진실 데이터 유사도", data.get('ts', 0), "#4CAF50"), unsafe_allow_html=True)
        with c2: st.markdown(colored_bar_html("가짜 데이터 유사도", data.get('fs', 0), "#F44336"), unsafe_allow_html=True)
        render_intelligence_distribution(prob)
        st.divider()
        st.write("###### [증거 3] 댓글 여론 분석")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("댓글 수", f"{data.get('cmt_count', 0)}")
        with c2: st.metric("주제 연관", data.get('cmt_rel', '-'))
        with c3: st.metric("선동 의심", f"{data.get('red_cnt', 0)}")
        if data.get('top_cmt_kw'): st.write(f"🗣️ **주요 키워드:** {', '.join(data['top_cmt_kw'])}")

    with tab_ai:
        st.write("###### [증거 4] AI 기술적 판단")
        st.info(f"**🤖 Logic:**\n{data.get('ai_reason', '판단 보류')}")
        st.write("###### 🔢 점수 산정 내역")
        if data.get('score_breakdown'): render_score_breakdown(data['score_breakdown'])

# --- [Execution Logic] ---
def run_forensic_main(url):
    st.session_state["debug_logs"] = []
    my_bar = st.progress(0, text="분석 엔진 가동 중...")
    db_count, _, _ = train_engine_wrapper()
    
    vid = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if not vid: st.error("URL 오류"); return
    vid = vid.group(1)

    with yt_dlp.YoutubeDL({'quiet':True, 'skip_download':True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            meta = {"제목":info.get('title'),"채널명":info.get('uploader'),"조회수":info.get('view_count',0),"댓글수":info.get('comment_count',0),"카테고리":", ".join(info.get('categories',[])),"해시태그":", ".join(info.get('tags',[])[:5])}
            
            my_bar.progress(20, "영상/자막 분석 중...")
            trans, _ = fetch_transcript(info)
            full_text = trans if trans else info.get('description', '')
            summary = full_text[:800] + "..."
            
            my_bar.progress(40, "키워드 추출 및 뉴스 검색...")
            queries, _ = get_keywords(meta['제목'], full_text)
            news_items, final_query = [], queries[0]
            for q in queries:
                items = fetch_news(q)
                if items: news_items=items; final_query=q; break
            
            my_bar.progress(60, "팩트체크 대조 분석...")
            news_ev, max_match = [], 0
            for item in news_items[:5]:
                s, r, src, r_url = verify_news(summary, item['link'], item['title'])
                if s > max_match: max_match = s
                icon = "🟢" if s>=80 else "🟡" if s>=60 else "🔴"
                news_ev.append({"뉴스 제목":item['title'],"일치도":f"{icon} {s}%","최종 점수":s,"분석 근거":r,"비고":src,"원문":r_url})
            
            cmts = fetch_comments(vid)
            top_kw, rel_score, rel_msg = analyze_comments(cmts, full_text)
            red_cnt, _ = check_red_flags(cmts)
            
            ts, fs = vector_engine.analyze(final_query+" "+meta['제목'])
            t_impact, f_impact = int(ts*30)*-1, int(fs*30)
            
            news_score = -40 if max_match>=80 else -15 if max_match>=70 else 10 if max_match>=60 else 30
            if not news_ev: news_score = 0
            if check_official(meta['채널명']): news_score = -50
            
            bait = 10 if count_agitation(meta['제목']) > 0 else -5
            base = 50 + t_impact + f_impact + news_score + min(20, red_cnt*3) + bait
            
            my_bar.progress(80, "AI 최종 판결...")
            ai_score, ai_reason = judge_final(meta['제목'], full_text, news_ev)
            final_prob = max(1, min(99, int(base*WEIGHT_ALGO + ai_score*WEIGHT_AI)))
            
            risk_text, _ = determine_risk_level(final_prob)
            final_summary = generate_comprehensive_summary(meta['제목'], final_prob, news_ev, red_cnt, ai_reason, risk_text)
            
            score_bd = [["기본 점수",50,"Base"],["진실 데이터 유사도",t_impact,"Vector"],["가짜 데이터 유사도",f_impact,"Vector"],["뉴스 팩트체크",news_score,"Fact"],["여론/어그로",min(20,red_cnt*3)+bait,"Sent"],["AI 판결",ai_score,"LLM"]]
            
            report = {"meta":meta,"summary":summary,"query_list":queries,"query":final_query,"score_breakdown":score_bd,"news_evidence":news_ev,"cmt_count":len(cmts),"cmt_rel":f"{rel_score}% ({rel_msg})","red_cnt":red_cnt,"top_cmt_kw":top_kw,"ai_reason":ai_reason,"ts":ts,"fs":fs,"final_summary":final_summary}
            
            save_db(meta['채널명'], meta['제목'], final_prob, url, final_query, report)
            my_bar.empty()
            render_report_full_ui(final_prob, db_count, meta['제목'], meta['채널명'], report, is_cached=False)
            
        except Exception as e: st.error(f"Error: {e}")

# --- [B2B Report] ---
def generate_b2b_report(df):
    if df.empty: return pd.DataFrame()
    df['fake_prob'] = pd.to_numeric(df['fake_prob'], errors='coerce').fillna(0)
    res = []
    for ch, g in df.groupby('channel_name'):
        avg = g['fake_prob'].mean()
        kws = []
        for k in g['keywords']:
            if isinstance(k, list): kws.extend([str(x) for x in k])
            elif k: kws.append(str(k))
        tokens = re.findall(r'[가-힣]{2,}', " ".join(kws))
        target = ", ".join([t[0] for t in Counter(tokens).most_common(3)])
        grade = "⛔ BLACKLIST" if avg>=60 else "⚠️ CAUTION" if avg>=40 else "✅ SAFE"
        res.append({"Channel":ch, "Grade":grade, "Avg Risk":f"{int(avg)}%", "Videos":len(g), "Target":target})
    return pd.DataFrame(res).sort_values("Avg Risk", ascending=False)

# --- [Layout Main] ---
st.title("⚖️유튜브 가짜뉴스 판독기 (Triple Engine)")

with st.container(border=True):
    with st.expander("ℹ️ 서비스 이용 안내 및 면책 조항 (Disclaimer)"):
        st.markdown("""
        본 서비스는 **인공지능(AI) 및 알고리즘 기반**으로 영상의 신뢰도를 분석하는 보조 도구입니다. 
        **분석 결과는 어떠한 법적 효력도 없으며, 최종 판단과 책임은 전적으로 사용자(당사자)에게 있습니다.**
        """)
    agree = st.checkbox("위 고지 내용을 확인하였으며, 결과에 대한 최종 책임이 본인에게 있음을 동의합니다.")

url = st.text_input("🔗 YouTube URL 입력")
if st.button("🚀 정밀 분석 시작", use_container_width=True, disabled=not agree):
    if url: run_forensic_main(url)
    else: st.warning("URL을 입력하세요.")

# --- [Bottom Section] ---
st.divider()
st.subheader("🗂️ DB History")

if st.session_state["is_admin"]:
    st.caption("✅ 관리자 모드: 삭제 가능")
else:
    st.caption("🔒 뷰어 모드: 조회만 가능")

try:
    # [수정] 테이블 이름을 'analysis_history'로 원상복구
    response = supabase.table("analysis_history").select("*").order("id", desc=True).execute()
    data = response.data
    if not data: st.info("📭 저장된 분석 기록이 없습니다.")
    else:
        df_hist = pd.DataFrame(data)
        if st.session_state["is_admin"]:
            if "Delete" not in df_hist.columns: df_hist.insert(0, "Delete", False)
            edited_df = st.data_editor(df_hist, hide_index=True, use_container_width=True, column_config={"Delete": st.column_config.CheckboxColumn("삭제", default=False), "fake_prob": st.column_config.NumberColumn("가짜 확률", format="%d%%"), "video_url": st.column_config.LinkColumn("URL"), "detail_json": None, "vector_json": None}, disabled=["id", "analysis_date", "channel_name", "video_title", "fake_prob", "keywords", "video_url"])
            if st.button("🗑️ 선택 항목 영구 삭제", type="primary"):
                to_delete = edited_df[edited_df['Delete'] == True]
                if not to_delete.empty:
                    # [수정] 테이블 이름을 'analysis_history'로 원상복구
                    for index, row in to_delete.iterrows(): supabase.table("analysis_history").delete().eq("id", row['id']).execute()
                    st.success("삭제 완료!"); time.sleep(1); st.rerun()
        else: st.dataframe(df_hist[['analysis_date','channel_name','video_title','fake_prob']], use_container_width=True, hide_index=True)
except Exception as e: st.error(f"❌ DB Error: {e}")

st.divider()
with st.expander("🔐 관리자 (Admin & B2B Report)"):
    if st.session_state["is_admin"]:
        st.success("Admin Logged In (Target: analysis_archive_v2)")
        
        st.write("### 🚑 데이터 이사 (최종_진짜_마지막.ver)")
        uploaded_file = st.file_uploader("백업 파일(export.csv)을 여기에 올리세요", type="csv")
        
        if uploaded_file is not None:
            if st.button("🚨 새 DB로 복구 시작", type="primary"):
                
                # 1. 파일 읽기
                try:
                    df_restore = pd.read_csv(uploaded_file)
                    st.info(f"📂 파일 읽기 성공: {len(df_restore)}개 데이터 대기 중...")
                except Exception as e:
                    st.error(f"파일 읽기 실패: {e}")
                    st.stop()

                restore_bar = st.progress(0)
                success_cnt = 0
                fail_cnt = 0
                
                # 2. 바로 데이터 주입 시작 (쓸데없는 조회 코드 삭제)
                for i, row in df_restore.iterrows():
                    # 제목 없는 데이터 패스
                    title = str(row.get('video_title', ''))
                    if title == 'nan' or not title: continue
                    
                    # 데이터 매핑
                    restore_data = {
                        "analysis_date": str(row.get('analysis_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))),
                        "channel_name": str(row.get('channel_name', 'Unknown')),
                        "video_title": title,
                        "fake_prob": int(row['fake_prob']) if pd.notna(row.get('fake_prob')) else 0,
                        "video_url": str(row.get('video_url', '')),
                        "keywords": str(row.get('keywords', '')),
                        "detail_json": {"final_summary": "복구된 데이터"},
                        "vector_json": None 
                    }
                    
                    try:
                        # 저장 시도
                        supabase.table("analysis_archive_v2").insert(restore_data).execute()
                        success_cnt += 1
                        
                    except Exception as e:
                        fail_cnt += 1
                        # 에러 발생 시 즉시 화면에 출력하고 멈춤
                        if fail_cnt == 1:
                            st.error(f"🚨 저장 실패! (첫 번째 데이터)")
                            st.error(f"에러 메시지: {e}")
                            st.write("넣으려고 했던 데이터:")
                            st.json(restore_data)
                            st.stop()
                    
                    restore_bar.progress(int(((i + 1) / len(df_restore)) * 100))
                
                # 결과
                st.write("---")
                if success_cnt > 0:
                    st.success(f"✅ {success_cnt}건 완벽하게 복구 성공!")
                    st.balloons()
                    st.info("이제 아래 [데이터 업데이트] 버튼을 눌러주세요!")
                else:
                    st.error("❌ 0건 저장됨.")

        st.write("---")

        # 3. 데이터 업데이트 (새 테이블 기준)
        st.write("### 🔧 시스템 관리")
        try:
            # 여기도 .head() 같은 거 안 쓰고 안전하게 조회
            res = supabase.table("analysis_archive_v2").select("id", count='exact').is_("vector_json", "null").execute()
            missing_count = res.count
        except: missing_count = 0

        if missing_count > 0:
            st.warning(f"⚠️ 학습 미반영 데이터 {missing_count}건")
            if st.button(f"♻️ 데이터 업데이트 ({missing_count}건)"):
                prog_text = st.empty()
                bar = st.progress(0)
                old_rows = supabase.table("analysis_archive_v2").select("*").is_("vector_json", "null").execute().data
                
                for i, row in enumerate(old_rows):
                    txt = f"{row.get('keywords','')} {row.get('video_title','')}"
                    try:
                        vec = vector_engine.get_embedding(txt)
                        supabase.table("analysis_archive_v2").update({"vector_json": vec}).eq("id", row['id']).execute()
                    except: continue
                    bar.progress(int(((i+1)/missing_count)*100))
                    prog_text.text(f"처리 중... {i+1}/{missing_count}")
                    time.sleep(0.5)
                st.success("완료!")
                time.sleep(1)
                st.rerun()
        else:
            st.success("✅ 모든 데이터가 최신 상태입니다.")

        if st.button("Logout"): st.session_state["is_admin"]=False; st.rerun()
    else:
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PASSWORD: st.session_state["is_admin"]=True; st.rerun()
            else: st.error("Wrong Password")


