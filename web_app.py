import streamlit as st
from supabase import create_client, Client
import re
import requests
import time
import random
import math
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI
from datetime import datetime
from collections import Counter
import yt_dlp
import pandas as pd
import altair as alt
import json
from bs4 import BeautifulSoup

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="유튜브 가짜뉴스 판독기 v99.8", layout="wide", page_icon="🛡️")

# 글로벌 상수
STATIC_TRUTH_CORPUS = ["박나래 위장전입 무혐의", "임영웅 암표 대응", "정희원 저속노화", "대전 충남 통합", "선거 출마 선언"]
STATIC_FAKE_CORPUS = ["충격 폭로 경악", "긴급 속보 소름", "충격 발언 논란", "구속 영장 발부", "영상 유출", "계시 예언", "사형 집행", "위독설"]
WEIGHT_ALGO = 0.6
WEIGHT_AI = 0.4
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']

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
    GOOGLE_API_KEY_A = st.secrets["GOOGLE_API_KEY_A"]
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
except Exception as e:
    st.error(f"❌ 필수 키 설정 누락: {e}")
    st.stop()

mistral_client = OpenAI(api_key=MISTRAL_API_KEY, base_url="https://api.mistral.ai/v1")

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [2. 유틸리티 & 파서 강화] ---
def parse_ai_json(text):
    if not text: return None
    try:
        # 마크다운 제거 로직 보강
        clean_text = re.sub(r'```json\s*', '', text)
        clean_text = re.sub(r'```', '', clean_text).strip()
        parsed = json.loads(clean_text)
        if isinstance(parsed, list) and len(parsed) > 0: return parsed[0]
        return parsed
    except:
        try:
            # 중괄호만 추출 시도
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match: return json.loads(match.group(1))
        except: pass
    return None

def safe_get_score(data_dict, default=50):
    """Mistral이 score 대신 '점수', 'fake_score' 등으로 보내도 찾아내는 함수"""
    if not data_dict: return default
    for key in ['score', '점수', 'fake_score', 'rating', 'value']:
        if key in data_dict: return int(float(data_dict[key]))
    return default

def safe_get_reason(data_dict, default="분석 결과 없음"):
    if not data_dict: return default
    for key in ['reason', '이유', '근거', '판단', 'analysis']:
        if key in data_dict: return data_dict[key]
    return default

def extract_video_id(url):
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else None

# --- [3. AI 모델 엔진] ---
@st.cache_data(ttl=3600)
def get_all_available_gemini_models(api_key):
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        models.sort(key=lambda x: 0 if 'lite' in x else 1 if 'flash' in x else 2)
        return models
    except: return ["models/gemini-1.5-flash"]

def get_gemini_search_keywords_survivor(title, transcript):
    genai.configure(api_key=GOOGLE_API_KEY_A)
    models = get_all_available_gemini_models(GOOGLE_API_KEY_A)
    prompt = f"Role: Fact-Check Investigator. [Input] Title: {title}, Transcript: {transcript[:15000]}. [Task] Extract ONE Korean search query (2-4 words). Output ONLY the string."
    for m in models:
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content(prompt)
            if response.text:
                st.session_state["debug_logs"].append(f"✅ Key A Success: {m}")
                return response.text.strip()
        except Exception as e:
            st.session_state["debug_logs"].append(f"❌ Key A Failed ({m}): {str(e)[:50]}")
            continue
    return title

def call_mistral_judge(prompt):
    try:
        response = mistral_client.chat.completions.create(
            model="mistral-large-latest",
            messages=[{"role": "system", "content": "당신은 전문 팩트체크 판사입니다. 반드시 한국어로 답변하고 JSON 형식({'score': int, 'reason': string})만 준수하세요."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        st.session_state["debug_logs"].append("✅ Key B (Mistral) Verdict Success")
        return response.choices[0].message.content
    except Exception as e:
        st.session_state["debug_logs"].append(f"❌ Key B Mistral Error: {e}")
        return None

# --- [4. 분석 엔진] ---
class VectorEngine:
    def __init__(self):
        self.vocab = set(); self.truth_vectors = []; self.fake_vectors = []
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
        if not self.vocab: return 0, 0
        qv = self.text_to_vector(query)
        mt = max([self.cosine_similarity(qv, v) for v in self.truth_vectors] or [0])
        mf = max([self.cosine_similarity(qv, v) for v in self.fake_vectors] or [0])
        return mt, mf

vector_engine = VectorEngine()

def fetch_comments_via_api(video_id):
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        res = requests.get(url, params={'part': 'snippet', 'videoId': video_id, 'key': YOUTUBE_API_KEY, 'maxResults': 50})
        items = [i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in res.json().get('items', [])]
        return items, "Success"
    except: return [], "Fail"

# --- [5. UI 컴포넌트] ---
def render_score_breakdown(data_list):
    style = """<style>table.score-table { width: 100%; border-collapse: separate; border: 1px solid #e0e0e0; border-radius: 8px; font-size: 14px; margin-top: 10px;} table.score-table th { background-color: #f8f9fa; padding: 12px; text-align: left; } table.score-table td { padding: 12px; border-bottom: 1px solid #f0f0f0; } .badge-danger { background-color: #ffebee; color: #d32f2f; padding: 4px 8px; border-radius: 4px; font-weight: bold; } .badge-success { background-color: #e8f5e9; color: #2e7d32; padding: 4px 8px; border-radius: 4px; font-weight: bold; }</style>"""
    rows = ""
    for item, score, note in data_list:
        try:
            val = int(score)
            badge = f'<span class="badge-danger">+{val}</span>' if val > 0 else f'<span class="badge-success">{val}</span>' if val < 0 else "0"
        except: badge = str(score)
        rows += f"<tr><td>{item}<br><small style='color:#888;'>{note}</small></td><td style='text-align:right;'>{badge}</td></tr>"
    st.markdown(f"{style}<table class='score-table'><thead><tr><th>분석 항목</th><th style='text-align:right;'>변동</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def colored_progress_bar(label, percent, color):
    st.markdown(f"""<div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between;"><span style="font-size: 13px; font-weight: 600;">{label}</span><span>{round(percent * 100, 1)}%</span></div><div style="background-color: #eee; height: 8px; border-radius: 5px;"><div style="background-color: {color}; height: 8px; width: {percent * 100}%; border-radius: 5px;"></div></div></div>""", unsafe_allow_html=True)

# --- [6. 메인 실행 함수] ---
def run_forensic_main(url):
    st.session_state["debug_logs"] = []
    vid = extract_video_id(url)
    if not vid: return st.error("URL 오류")

    # DB 로드 & 학습
    res_t = supabase.table("analysis_history").select("video_title").lt("fake_prob", 40).execute()
    res_f = supabase.table("analysis_history").select("video_title").gt("fake_prob", 60).execute()
    dt, df = [r['video_title'] for r in res_t.data], [r['video_title'] for r in res_f.data]
    vector_engine.train(STATIC_TRUTH_CORPUS + dt, STATIC_FAKE_CORPUS + df)
    db_count = len(dt) + len(df)

    # 캐시 체크
    cached_res = supabase.table("analysis_history").select("*").ilike("video_url", f"%{vid}%").order("id", desc=True).limit(1).execute()
    if cached_res.data:
        c = cached_res.data[0]
        try:
            d = json.loads(c.get('detail_json', '{}'))
            render_report_full_ui(c['fake_prob'], db_count, c['video_title'], c['channel_name'], d, is_cached=True)
            return
        except: pass

    my_bar = st.progress(0, text="분석 시작...")
    with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title, uploader, desc = info.get('title',''), info.get('uploader',''), info.get('description','')
            
            # 자막/댓글 수집
            my_bar.progress(10, "1단계: 데이터 수집 중...")
            subs = info.get('subtitles') or {}; auto = info.get('automatic_captions') or {}; merged = {**subs, **auto}
            full_text = desc
            if 'ko' in merged:
                for f in merged['ko']:
                    if f['ext'] == 'vtt':
                        full_text = " ".join([l.strip() for l in requests.get(f['url']).text.splitlines() if l.strip() and '-->' not in l and '<' not in l])
                        break
            cmts, _ = fetch_comments_via_api(vid)

            # Key A
            my_bar.progress(30, "2단계: 키워드 추출 중...")
            query = get_gemini_search_keywords_survivor(title, full_text)
            
            # Key B 뉴스 대조
            my_bar.progress(50, "3단계: 뉴스 대조 중...")
            rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
            items = re.findall(r'<item>(.*?)</item>', requests.get(rss).text, re.DOTALL)[:3]
            news_ev = []; max_match = 0
            for i in items:
                nt = re.search(r'<title>(.*?)</title>', i).group(1).replace("<![CDATA[", "").replace("]]>", "")
                nl = re.search(r'<link>(.*?)</link>', i).group(1)
                nd = re.search(r'<description>(.*?)</description>', i).group(1)
                
                res_b = call_mistral_judge(f"영상[{title}] vs 뉴스[{nt}]. 일치여부 판단. JSON {{'score', 'reason'}}")
                p_b = parse_ai_json(res_b)
                s_b = safe_get_score(p_b, 50)
                if s_b > max_match: max_match = s_b
                news_ev.append({"뉴스 제목": nt, "일치도": f"{s_b}%", "최종 점수": s_b, "분석 근거": safe_get_reason(p_b), "원문": nl})

            # 점수 계산
            ts, fs = vector_engine.analyze_position(query + " " + title)
            t_impact, f_impact = int(ts*30)*-1, int(fs*30)
            news_penalty = -30 if max_match <= 20 else (30 if max_match >= 80 else 0)
            
            # 최종 판결
            my_bar.progress(85, "4단계: AI 최종 판결 중...")
            res_f = call_mistral_judge(f"영상 '{title}', 뉴스 증거: {news_ev}. 진실 0-20, 가짜 80-100. JSON {{'score', 'reason'}}")
            p_f = parse_ai_json(res_f)
            ai_score = safe_get_score(p_f, 50)
            ai_reason = safe_get_reason(p_f)
            
            final_prob = max(1, min(99, int((50 + t_impact + f_impact + news_penalty)*WEIGHT_ALGO + ai_score*WEIGHT_AI)))
            
            score_breakdown = [["기본 점수", 50, "중립 시작"], ["진실 DB 매칭", t_impact, ""], ["가짜 패턴 매칭", f_impact, ""], ["뉴스 교차 검증", news_penalty, ""], ["AI 판결 점수", ai_score, ai_reason]]
            
            report = {
                "summary": full_text[:800], "news_evidence": news_ev, "ai_score": ai_score, "ai_reason": ai_reason,
                "score_breakdown": score_breakdown, "ts": ts, "fs": fs, "query": query, "cmt_count": len(cmts)
            }
            
            supabase.table("analysis_history").insert({"channel_name": uploader, "video_title": title, "fake_prob": final_prob, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url": url, "keywords": query, "detail_json": json.dumps(report, ensure_ascii=False)}).execute()
            my_bar.empty()
            render_report_full_ui(final_prob, db_count, title, uploader, report)

        except Exception as e: st.error(f"오류: {e}")

def render_report_full_ui(prob, db_count, title, uploader, d, is_cached=False):
    if is_cached: st.success("🎉 기존 분석 데이터 로드")
    st.subheader("🕵️ Dual-Engine Analysis Result")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("가짜뉴스 확률", f"{prob}%")
    col_b.metric("AI 판정", "🔴 위험" if prob > 60 else "🟢 안전" if prob < 30 else "🟠 주의")
    col_c.metric("지능 노드", f"{db_count} Nodes")
    
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.write(f"**제목:** {title}\n**채널:** {uploader}")
        st.info(f"🎯 검색어: {d.get('query', 'N/A')}")
        render_score_breakdown(d.get('score_breakdown', []))
    with col2:
        st.write("📊 **5대 정밀 분석 증거**")
        colored_progress_bar("✅ 진실 영역 근접도", d.get('ts', 0), "#2ecc71")
        colored_progress_bar("🚨 거짓 영역 근접도", d.get('fs', 0), "#e74c3c")
        st.markdown("**[증거 1] 뉴스 교차 대조**")
        st.dataframe(pd.DataFrame(d.get('news_evidence', [])), use_container_width=True, hide_index=True)
        with st.container(border=True): st.write(f"⚖️ **AI 판결:** {d.get('ai_reason', 'N/A')}")

# --- [7. UI 레이아웃] ---
st.title("⚖️ Fact-Check Center v99.8")
with st.container(border=True):
    st.markdown("### 🛡️ 법적 고지")
    agree = st.checkbox("내용을 확인했으며 분석에 동의합니다.")

url_input = st.text_input("🔗 URL")
if st.button("🚀 분석 시작", disabled=not agree): run_forensic_main(url_input)

st.divider()
try:
    resp = supabase.table("analysis_history").select("*").order("id", desc=True).limit(10).execute()
    df = pd.DataFrame(resp.data)
    if not df.empty:
        if st.session_state["is_admin"]:
            df['Delete'] = False
            edited = st.data_editor(df[['Delete', 'id', 'video_title', 'fake_prob', 'keywords']], hide_index=True, use_container_width=True)
            if st.button("🗑️ 삭제"):
                for _, row in edited[edited.Delete].iterrows(): supabase.table("analysis_history").delete().eq("id", row['id']).execute()
                st.rerun()
        else: st.dataframe(df[['analysis_date', 'video_title', 'fake_prob', 'keywords']], use_container_width=True, hide_index=True)
except: pass

with st.expander("🔐 관리자 전용"):
    if not st.session_state["is_admin"]:
        if st.text_input("PW", type="password") == ADMIN_PASSWORD: st.session_state["is_admin"] = True; st.rerun()
    else:
        st.write(f"🤖 하이브리드 엔진 (Gemini A + Mistral B) Active")
        if st.session_state["debug_logs"]: st.text_area("Debug Logs", "\n".join(st.session_state["debug_logs"]), height=300)
        if st.button("Logout"): st.session_state["is_admin"] = False; st.rerun()
