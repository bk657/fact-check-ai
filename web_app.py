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

# --- [1. 시스템 설정 및 CSS 최적화] ---
st.set_page_config(page_title="유튜브 가짜뉴스 판독기 (Triple Engine)", layout="wide", page_icon="🛡️")

# [Mobile/Web UI 최적화 CSS]
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

# --- [3. 유틸리티] ---
def parse_llm_json(text):
    try:
        parsed = json.loads(text)
    except:
        try:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```', '', text)
            match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if match: parsed = json.loads(match.group(1))
            else: return None
        except: return None
    if isinstance(parsed, list): return parsed[0] if len(parsed) > 0 and isinstance(parsed[0], dict) else None
    if isinstance(parsed, dict): return parsed
    return None

def determine_risk_level(prob):
    if prob >= 70: return "⛔ 위험 (High Risk)", "#d32f2f" # Red
    elif prob >= 40: return "⚠️ 주의 (Caution)", "#f57c00" # Orange
    return "✅ 안전 (Safe)", "#388e3c" # Green

def colored_bar_html(label, score, color):
    pct = min(100, max(0, int(score * 100)))
    return f"""
    <div style="margin-bottom: 6px;">
        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: #444;">
            <span>{label}</span>
            <span>{pct}%</span>
        </div>
        <div style="width: 100%; background-color: #e0e0e0; border-radius: 6px; height: 8px; margin-top: 2px;">
            <div style="width: {pct}%; background-color: {color}; height: 8px; border-radius: 6px;"></div>
        </div>
    </div>
    """

# --- [4. Core Logic] ---
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

    # Gemini A & B
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

# --- [5. Data & Engine] ---
WEIGHT_ALGO = 0.85
WEIGHT_AI = 0.15
OFFICIAL_CHANNELS = ['MBC','KBS','SBS','EBS','YTN','JTBC','TVCHOSUN','MBN','CHANNEL A','연합뉴스','YONHAP','한겨레','경향','조선','중앙','동아']
CRITICAL_STATE_KEYWORDS = ['별거','이혼','파경','사망','위독','구속','체포','실형','불화','폭로','충격','논란','심정지','뇌사','압수수색','감옥']
STATIC_TRUTH = ["박나래 위장전입 무혐의", "임영웅 암표 대응", "정희원 저속노화", "선거 출마 선언"]
STATIC_FAKE = ["충격 폭로 경악", "긴급 속보 소름", "구속 영장 발부", "사형 집행", "위독설"]

class VectorEngine:
    def __init__(self): self.vocab=set(); self.truth=[]; self.fake=[]
    def tokenize(self, t): return re.findall(r'[가-힣]{2,}', t)
    def train(self, t_list, f_list):
        for t in t_list+f_list: self.vocab.update(self.tokenize(t))
        self.vocab = sorted(list(self.vocab))
        self.truth = [self.vec(t) for t in t_list]
        self.fake = [self.vec(t) for t in f_list]
    def vec(self, t):
        c = Counter(self.tokenize(t))
        return [c[w] for w in self.vocab]
    def sim(self, v1, v2):
        dot = sum(a*b for a,b in zip(v1,v2))
        mag = math.sqrt(sum(a*a for a in v1))*math.sqrt(sum(b*b for b in v2))
        return dot/mag if mag>0 else 0
    def analyze(self, q):
        qv = self.vec(q)
        return max([self.sim(qv,v) for v in self.truth] or [0]), max([self.sim(qv,v) for v in self.fake] or [0])

vector_engine = VectorEngine()

# [수정] 3-Way 전략 명시 및 키워드 추출
def get_keywords(title, trans):
    prompt = f"""
    You are a Fact-Check Investigator.
    [Input] Title: {title}, Transcript: {trans[:10000]}
    [Task] Generate 3 diverse Google News search queries to verify this video.
    1. Specific: Entity + Exact Event (Specific Incident)
    2. Broader: Main Subject + Status (Contextual)
    3. Keywords: Core Nouns Combination
    
    [Output JSON] {{ "queries": ["query1", "query2", "query3"] }}
    """
    res, model, logs = call_triple_survivor(prompt, is_json=True)
    st.session_state["debug_logs"].extend([f"[Key] {l}" for l in logs])
    parsed = parse_llm_json(res)
    # 파싱 성공 시 쿼리 리스트 반환, 실패 시 제목 그대로 사용
    if parsed and 'queries' in parsed and isinstance(parsed['queries'], list):
        return parsed['queries'], model
    return [title, title + " 뉴스", title + " 팩트체크"], model

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
    prompt = f"""
    당신은 팩트체크 전문 AI 분석가입니다. 아래 데이터를 바탕으로 사용자에게 최종 종합 리포트를 작성해주세요.
    
    [분석 데이터]
    - 영상 제목: {title}
    - 최종 가짜뉴스 확률: {final_prob}% ({risk_text})
    - 뉴스 대조 결과: {len(news_ev)}개의 기사와 대조됨
    - 선동성 댓글 감지: {red_cnt}개
    - AI 판단 요약: {ai_reason}
    
    [요청사항]
    1. 이 영상이 왜 {final_prob}% 점수를 받았는지 핵심 이유를 요약하세요.
    2. 뉴스 증거와의 일치 여부, 제목의 어그로성, 여론 반응을 종합적으로 언급하세요.
    3. 사용자에게 "믿어도 되는지", "주의해야 하는지" 명확한 행동 가이드를 제시하세요.
    4. 한국어로 정중하고 전문적인 어조로 작성하세요. (최대 4문장)
    """
    res, _, _ = call_triple_survivor(prompt, is_json=False)
    return res if res else "종합 분석 결과를 생성하는데 실패했습니다."

# --- [6. Helper Functions] ---
def normalize(w): return re.sub(r'은$|는$|이$|가$|을$|를$|의$|에$|로$', '', re.sub(r'[^가-힣0-9]', '', w))
def get_tokens(t): return [normalize(w) for w in re.findall(r'[가-힣]{2,}', t) if w not in ['충격','속보','뉴스']]
def get_top_kw(t): return Counter(get_tokens(t)).most_common(5)
def check_official(n): return any(o in n.upper().replace(" ","") for o in OFFICIAL_CHANNELS)
def count_agitation(t): return sum(t.count(w) for w in ['충격','경악','실체','폭로','속보','소름'])
def check_red_flags(cmts): 
    d = [k for c in cmts for k in ['가짜','주작','구라','허위','선동'] if k in c]
    return len(d), list(set(d))

# --- [Data Fetching] ---
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

def save_db(ch, ti, pr, url, kw, detail):
    try: supabase.table("analysis_history").insert({
        "channel_name":ch, "video_title":ti, "fake_prob":pr, "video_url":url, 
        "keywords":kw, "detail_json":json.dumps(detail, ensure_ascii=False),
        "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }).execute()
    except Exception as e: print(f"DB Error: {e}")

# --- [UI 렌더링 함수 (Conclusion First)] ---
def render_report_full_ui(prob, db_count, title, channel, data, is_cached=False):
    st.divider()
    if is_cached: st.info(f"💾 과거 분석 기록 호출됨 (총 DB 데이터: {db_count}개)")
    
    risk_text, risk_color = determine_risk_level(prob)
    
    # 1. [HERO SECTION] Score & Risk (최상단)
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1: st.metric("🔥 가짜뉴스 확률", f"{prob}%")
    with c2: st.markdown(f"<div style='text-align:center; padding:10px; border-radius:10px; background-color:{risk_color}; color:white; font-weight:bold; font-size:1.1rem; margin-top:5px;'>{risk_text}</div>", unsafe_allow_html=True)
    with c3: st.metric("🗄️ 누적 DB", f"{db_count}건")
    
    # 2. [FINAL SUMMARY] AI 종합 리포트 (바로 아래 배치)
    st.subheader("📝 AI 최종 종합 리포트")
    with st.container(border=True):
        st.markdown(f"**📢 AI Analyst Comment:**\n\n{data.get('final_summary', '분석 데이터가 생성되지 않았습니다.')}")

    # 3. [VIDEO INFO] 영상 기본 정보
    st.subheader("ℹ️ 영상 기본 정보")
    with st.container(border=True):
        st.write(f"**📺 {title}**")
        st.caption(f"채널: {channel} | 분석일: {datetime.now().strftime('%Y-%m-%d')}")
        st.info(f"**내용 요약:** {data.get('summary', '요약 없음')}")
        with st.expander("상세 메타데이터 보기"):
            st.dataframe(pd.DataFrame([data.get('meta', {})]), use_container_width=True, hide_index=True)

    # 4. [EVIDENCE TABS] 상세 증거 자료 (탭으로 분리)
    st.subheader("🔍 상세 증거 및 분석 데이터")
    tab_news, tab_data, tab_ai = st.tabs(["📰 뉴스 팩트체크", "📊 데이터/여론", "🤖 AI 기술적 판단"])
    
    # [Tab 1: News Check]
    with tab_news:
        # [NEW] 검색 키워드 정보 표시 (3-Way)
        st.markdown("###### 🗝️ AI 검색 키워드 (3-Way Strategy)")
        if data.get('query_list'):
            # 보기 좋게 포맷팅
            q_list_formatted = " | ".join([f"`{q}`" for q in data['query_list']])
            st.caption(f"AI가 추출한 3가지 전략 키워드:\n{q_list_formatted}")
        
        if data.get('query'):
            st.success(f"✅ 뉴스 검색에 성공한 최종 키워드: **{data['query']}**")
        
        st.divider()
        st.write("###### [증거 2] 주요 뉴스 대조 결과 (Top 5)")
        if data.get('news_evidence'):
            for news in data['news_evidence']:
                with st.expander(f"{news['일치도']} {news['뉴스 제목']}"):
                    st.write(f"**🕵️ 분석 근거:** {news['분석 근거']}")
                    st.caption(f"출처: {news['비고']}")
                    st.link_button("🔗 기사 원문 보기", news['원문'])
        else:
            st.warning("관련된 신뢰할 수 있는 뉴스 기사를 찾지 못했습니다.")

    # [Tab 2: Data & Sentiment]
    with tab_data:
        st.write("###### [증거 1] 데이터 유사도 분석")
        c1, c2 = st.columns(2)
        with c1: st.markdown(colored_bar_html("진실 데이터 유사도", data.get('ts', 0), "#4CAF50"), unsafe_allow_html=True)
        with c2: st.markdown(colored_bar_html("가짜 데이터 유사도", data.get('fs', 0), "#F44336"), unsafe_allow_html=True)
        
        st.caption("※ 전체 DB 분포 내 현재 영상 위치")
        render_intelligence_distribution(prob)
        
        st.divider()
        st.write("###### [증거 3] 댓글 여론 분석")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1: st.metric("댓글 수", f"{data.get('cmt_count', 0)}")
        with col_c2: st.metric("주제 연관성", data.get('cmt_rel', '-'))
        with col_c3: st.metric("선동 의심 댓글", f"{data.get('red_cnt', 0)}")
        
        if data.get('top_cmt_kw'):
            st.write(f"🗣️ **주요 키워드:** {', '.join(data['top_cmt_kw'])}")

    # [Tab 3: AI Logic]
    with tab_ai:
        st.write("###### [증거 4] AI 기술적 판단 로직")
        st.info(f"**🤖 Internal Logic:**\n{data.get('ai_reason', '판단 보류')}")
        
        st.write("###### 🔢 점수 산정 내역 (Score Breakdown)")
        if data.get('score_breakdown'):
            render_score_breakdown(data['score_breakdown'])


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

# --- [Main Analysis Logic] ---
def run_forensic_main(url):
    st.session_state["debug_logs"] = []
    my_bar = st.progress(0, text="분석 엔진 가동 중...")
    
    # 0. 학습 데이터 로드
    db_count, dt, df = train_engine_wrapper()
    
    vid = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if not vid: st.error("올바른 유튜브 URL이 아닙니다."); return
    vid = vid.group(1)

    with yt_dlp.YoutubeDL({'quiet':True, 'skip_download':True}) as ydl:
        try:
            # 1. 메타 데이터 수집
            info = ydl.extract_info(url, download=False)
            meta = {
                "제목": info.get('title'),
                "채널명": info.get('uploader'),
                "조회수": info.get('view_count', 0),
                "댓글수": info.get('comment_count', 0),
                "카테고리": ", ".join(info.get('categories', [])),
                "해시태그": ", ".join(info.get('tags', [])[:5])
            }
            
            my_bar.progress(20, "영상 내용 분석 중...")
            trans, _ = fetch_transcript(info)
            full_text = trans if trans else info.get('description', '')
            summary = full_text[:800] + "..."
            
            my_bar.progress(40, "키워드 추출 및 뉴스 검색 중...")
            queries, _ = get_keywords(meta['제목'], full_text)
            
            news_items = []
            final_query = queries[0]
            for q in queries:
                items = fetch_news(q)
                if items: news_items = items; final_query = q; break
            
            my_bar.progress(60, "팩트체크 대조 분석 중...")
            # [수정] 5개까지 검증
            news_ev = []; max_match = 0
            for item in news_items[:5]:
                s, r, src, r_url = verify_news(summary, item['link'], item['title'])
                if s > max_match: max_match = s
                icon = "🟢" if s>=80 else "🟡" if s>=60 else "🔴"
                news_ev.append({"뉴스 제목":item['title'], "일치도":f"{icon} {s}%", "최종 점수":s, "분석 근거":r, "비고":src, "원문":r_url})
            
            cmts = fetch_comments(vid)
            top_kw, rel_score, rel_msg = analyze_comments(cmts, full_text)
            red_cnt, _ = check_red_flags(cmts)
            
            ts, fs = vector_engine.analyze(final_query + " " + meta['제목'])
            t_impact, f_impact = int(ts*30)*-1, int(fs*30)
            
            news_score = -40 if max_match>=80 else -15 if max_match>=70 else 10 if max_match>=60 else 30
            if not news_ev: news_score = 0
            if check_official(meta['채널명']): news_score = -50
            
            agitation = count_agitation(meta['제목'])
            bait = 10 if agitation > 0 else -5
            
            base_score = 50 + t_impact + f_impact + news_score + min(20, red_cnt*3) + bait
            
            my_bar.progress(80, "AI 최종 판결 중...")
            ai_score, ai_reason = judge_final(meta['제목'], full_text, news_ev)
            
            final_prob = max(1, min(99, int(base_score*WEIGHT_ALGO + ai_score*WEIGHT_AI)))
            
            # [추가] 최종 종합 리포트 생성
            risk_text, _ = determine_risk_level(final_prob)
            final_summary = generate_comprehensive_summary(meta['제목'], final_prob, news_ev, red_cnt, ai_reason, risk_text)

            score_bd = [
                ["기본 점수", 50, "Base Score"],
                ["진실 데이터 유사도", t_impact, "Truth Corpus Similarity"],
                ["가짜 데이터 유사도", f_impact, "Fake Corpus Similarity"],
                ["뉴스 팩트체크", news_score, "Journalism Match"],
                ["여론 및 어그로", min(20, red_cnt*3) + bait, "Sentiment & Clickbait"],
                ["AI 판결 (가중치)", ai_score, "LLM Judge"]
            ]
            
            report = {
                "meta": meta, "summary": summary, "query_list": queries, "query": final_query,
                "score_breakdown": score_bd, "news_evidence": news_ev,
                "cmt_count": len(cmts), "cmt_rel": f"{rel_score}% ({rel_msg})", "red_cnt": red_cnt, "top_cmt_kw": top_kw,
                "ai_reason": ai_reason, "ts": ts, "fs": fs,
                "final_summary": final_summary # 저장
            }
            
            save_db(meta['채널명'], meta['제목'], final_prob, url, final_query, report)
            my_bar.empty()
            render_report_full_ui(final_prob, db_count, meta['제목'], meta['채널명'], report, is_cached=False)
            
        except Exception as e: st.error(f"Error: {e}")

def train_engine_wrapper():
    try:
        res_t = supabase.table("analysis_history").select("video_title").lt("fake_prob", 40).execute()
        res_f = supabase.table("analysis_history").select("video_title").gt("fake_prob", 60).execute()
        dt = [r['video_title'] for r in res_t.data] if res_t.data else []
        df = [r['video_title'] for r in res_f.data] if res_f.data else []
        vector_engine.train(STATIC_TRUTH + dt, STATIC_FAKE + df)
        return len(dt)+len(df), dt, df
    except:
        vector_engine.train(STATIC_TRUTH, STATIC_FAKE)
        return 0, [], []

# --- [B2B Report Logic] ---
def generate_b2b_report(df):
    if df.empty: return pd.DataFrame()
    df['fake_prob'] = pd.to_numeric(df['fake_prob'], errors='coerce').fillna(0)
    res = []
    for ch, g in df.groupby('channel_name'):
        avg = g['fake_prob'].mean()
        # 키워드 flatten
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
        
        * **1st Line**: Mistral AI (Logic Analysis)
        * **2nd Line**: Google Gemini (Cross-Check)
        * **3rd Line**: Deep News Crawler (Fact Verification)
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
    response = supabase.table("analysis_history").select("*").order("id", desc=True).execute()
    data = response.data
    
    if not data:
        st.info("📭 저장된 분석 기록이 없습니다.")
    else:
        df_hist = pd.DataFrame(data)
        
        if st.session_state["is_admin"]:
            if "Delete" not in df_hist.columns:
                df_hist.insert(0, "Delete", False)
            
            edited_df = st.data_editor(
                df_hist,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Delete": st.column_config.CheckboxColumn("삭제", help="체크 후 삭제 버튼 클릭", default=False),
                    "fake_prob": st.column_config.NumberColumn("가짜 확률", format="%d%%"),
                    "video_url": st.column_config.LinkColumn("URL"),
                    "detail_json": None 
                },
                disabled=["id", "analysis_date", "channel_name", "video_title", "fake_prob", "keywords", "video_url"]
            )
            
            if st.button("🗑️ 선택 항목 영구 삭제", type="primary"):
                to_delete = edited_df[edited_df['Delete'] == True]
                if not to_delete.empty:
                    for index, row in to_delete.iterrows():
                        supabase.table("analysis_history").delete().eq("id", row['id']).execute()
                    st.success(f"{len(to_delete)}개 항목 삭제 완료!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("삭제할 항목을 선택해주세요.")
        else:
            st.dataframe(
                df_hist[['analysis_date','channel_name','video_title','fake_prob']], 
                use_container_width=True, 
                hide_index=True
            )
except Exception as e:
    st.error(f"❌ DB 불러오기 실패: {e}")

st.divider()
with st.expander("🔐 관리자 (Admin & B2B Report)"):
    if st.session_state["is_admin"]:
        st.success("Admin Logged In")
        if st.button("📊 B2B 리포트 생성"):
            try:
                rpt = generate_b2b_report(pd.DataFrame(data))
                if not rpt.empty:
                    st.dataframe(rpt, use_container_width=True)
                    st.download_button("📥 CSV 다운로드", rpt.to_csv().encode('utf-8-sig'), "b2b_report.csv", "text/csv")
            except: st.error("데이터 부족")
        
        st.write("📜 System Logs")
        st.text_area("Logs", "\n".join(st.session_state["debug_logs"]), height=200)
        
        if st.button("Logout"): st.session_state["is_admin"]=False; st.rerun()
    else:
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PASSWORD: st.session_state["is_admin"]=True; st.rerun()
            else: st.error("Wrong Password")
