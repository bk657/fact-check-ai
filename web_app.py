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
st.set_page_config(page_title="Fact-Check Center v97.0", layout="wide", page_icon="⚖️")

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
    GOOGLE_API_KEY_B = st.secrets["GOOGLE_API_KEY_B"]
except:
    st.error("❌ 필수 키(API Keys)가 설정되지 않았습니다.")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [2. 유틸리티 & JSON 파서] ---
def parse_gemini_json(text):
    try:
        return json.loads(text)
    except:
        try:
            text = re.sub(r'```json\s*', '', text).replace('```', '')
            match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
                return parsed[0] if isinstance(parsed, list) else parsed
        except: pass
    return None

def extract_video_id(url):
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else None

# --- [3. 모델 자동 탐색기] ---
@st.cache_data(ttl=3600)
def get_all_available_models(api_key):
    genai.configure(api_key=api_key)
    try:
        models = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        models.sort(key=lambda x: 0 if 'lite' in x else 1 if 'flash' in x else 2)
        return models
    except:
        return ["gemini-2.5-flash-lite", "gemini-2.0-flash"]

# --- [4. 상수 및 벡터 엔진] ---
WEIGHT_ALGO = 0.6
WEIGHT_AI = 0.4
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', '채널A', 'TV조선', '연합뉴스', '한겨레', '조선일보', '중앙일보', '동아일보']
STATIC_TRUTH_CORPUS = ["박나래 위장전입 무혐의", "임영웅 암표 대응", "정희원 저속노화", "대전 충남 통합", "선거 출마 선언"]
STATIC_FAKE_CORPUS = ["충격 폭로 경악", "긴급 속보 소름", "충격 발언 논란", "구속 영장 발부", "영상 유출", "계시 예언", "사형 집행", "위독설"]

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
        if not self.vocab: return 0, 0
        qv = self.text_to_vector(query)
        mt = max([self.cosine_similarity(qv, v) for v in self.truth_vectors] or [0])
        mf = max([self.cosine_similarity(qv, v) for v in self.fake_vectors] or [0])
        return mt, mf

vector_engine = VectorEngine()

# --- [5. Gemini Logic (Survivor)] ---
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
            time.sleep(0.1)
            continue
    return None, "All Failed", logs

# --- [6. UI 및 분석 유틸리티] ---
def render_score_breakdown(data_list):
    style = """<style>table.score-table { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; font-family: sans-serif; font-size: 14px; margin-top: 10px;} table.score-table th { background-color: #f8f9fa; color: #495057; font-weight: bold; padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; } table.score-table td { padding: 12px 15px; border-bottom: 1px solid #f0f0f0; color: #333; } .badge { padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-align: center; min-width: 45px; } .badge-danger { background-color: #ffebee; color: #d32f2f; } .badge-success { background-color: #e8f5e9; color: #2e7d32; } .badge-neutral { background-color: #f5f5f5; color: #757575; border: 1px solid #e0e0e0; }</style>"""
    rows = ""
    for item, score, note in data_list:
        try:
            score_num = int(score)
            badge = f'<span class="badge badge-danger">+{score_num}</span>' if score_num > 0 else f'<span class="badge badge-success">{score_num}</span>' if score_num < 0 else f'<span class="badge badge-neutral">0</span>'
        except: badge = f'<span class="badge badge-neutral">{score}</span>'
        rows += f"<tr><td>{item}<br><span style='color:#888; font-size:11px;'>{note}</span></td><td style='text-align: right;'>{badge}</td></tr>"
    st.markdown(f"{style}<table class='score-table'><thead><tr><th>분석 항목</th><th style='text-align: right;'>변동</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def colored_progress_bar(label, percent, color):
    st.markdown(f"""<div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span style="font-size: 13px; font-weight: 600; color: #555;">{label}</span><span style="font-size: 13px; font-weight: 700; color: {color};">{round(percent * 100, 1)}%</span></div><div style="background-color: #eee; border-radius: 5px; height: 8px; width: 100%;"><div style="background-color: {color}; height: 8px; width: {percent * 100}%; border-radius: 5px;"></div></div></div>""", unsafe_allow_html=True)

# --- [7. 메인 로직] ---
def run_forensic_main(url):
    st.session_state["debug_logs"] = []
    vid = extract_video_id(url)
    if not vid: return st.error("URL 오류")

    # DB 로드 및 벡터 학습
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
            st.success("🎉 기존 분석 데이터 로드")
            # [UI 리포트 출력 섹션 - 아래 레이아웃과 동일]
            render_report_ui(c['fake_prob'], db_count, c['video_title'], c['channel_name'], d, is_cached=True)
            return
        except: pass

    my_bar = st.progress(0, text="분석 시작...")
    with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title, uploader, desc = info.get('title',''), info.get('uploader',''), info.get('description','')
            
            # 자막 수집
            subs = info.get('subtitles') or {}
            auto = info.get('automatic_captions') or {}
            merged = {**subs, **auto}
            full_text = desc
            if 'ko' in merged:
                for f in merged['ko']:
                    if f['ext'] == 'vtt':
                        res = requests.get(f['url'])
                        full_text = " ".join([l.strip() for l in res.text.splitlines() if l.strip() and '-->' not in l and '<' not in l])
                        break

            # AI 수사관 (Key A)
            query_res, model_a, logs_a = call_gemini_survivor(GOOGLE_API_KEY_A, f"Extract 1 Korean News Query for: {title}, {full_text[:5000]}")
            st.session_state["debug_logs"].extend(logs_a)
            query = query_res.strip() if query_res else title

            # 뉴스 크롤링 및 AI 판사 (Key B)
            rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
            items = re.findall(r'<item>(.*?)</item>', requests.get(rss).text, re.DOTALL)[:3]
            news_ev = []; max_match = 0
            for i in items:
                nt = re.search(r'<title>(.*?)</title>', i).group(1)
                nl = re.search(r'<link>(.*?)</link>', i).group(1)
                nd = re.search(r'<description>(.*?)</description>', i).group(1)
                # 뉴스 본문 크롤링 시도
                txt, _ = scrape_news_content_robust(nl)
                score_b, reason_b, logs_b = call_gemini_survivor(GOOGLE_API_KEY_B, f"Compare {title} vs {txt if txt else nd}. JSON {{score, reason}}", is_json=True)
                st.session_state["debug_logs"].extend(logs_b)
                p_b = parse_gemini_json(score_b)
                sb = p_b.get('score', 50) if p_b else 50
                if sb > max_match: max_match = sb
                news_ev.append({"뉴스 제목": nt, "일치도": f"{sb}%", "분석 근거": p_b.get('reason','') if p_b else 'N/A', "원문": nl})

            # 최종 점수 계산
            ts, fs = vector_engine.analyze_position(query + " " + title)
            t_impact, f_impact = int(ts*30)*-1, int(fs*30)
            news_penalty = -30 if max_match <= 20 else (30 if max_match >= 80 else 0)
            
            ai_score_res, model_final, logs_final = call_gemini_survivor(GOOGLE_API_KEY_B, f"Final Verdict for {title}. News: {news_ev}. JSON {{score, reason}}", is_json=True)
            st.session_state["debug_logs"].extend(logs_final)
            p_final = parse_gemini_json(ai_score_res)
            ai_score = p_final.get('score', 50) if p_final else 50
            
            final_prob = max(1, min(99, int((50 + t_impact + f_impact + news_penalty)*WEIGHT_ALGO + ai_score*WEIGHT_AI)))
            
            report = {
                "summary": full_text[:500], "news_evidence": news_ev, "ai_score": ai_score, "ai_reason": p_final.get('reason','') if p_final else 'Error',
                "score_breakdown": [["기본 점수", 50, "중립"], ["DB 진실", t_impact, ""], ["DB 거짓", f_impact, ""], ["뉴스 검증", news_penalty, ""], ["AI 판결", ai_score, ""]],
                "ts": ts, "fs": fs, "query": query
            }
            
            supabase.table("analysis_history").insert({"channel_name": uploader, "video_title": title, "fake_prob": final_prob, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url": url, "keywords": query, "detail_json": json.dumps(report, ensure_ascii=False)}).execute()
            my_bar.empty()
            render_report_ui(final_prob, db_count, title, uploader, report)

        except Exception as e: st.error(f"오류: {e}")

def render_report_ui(prob, db_count, title, uploader, d, is_cached=False):
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
        st.write("📊 **정밀 분석 증거**")
        colored_progress_bar("✅ 진실 유사도", d.get('ts', 0), "#2ecc71")
        colored_progress_bar("🚨 거짓 유사도", d.get('fs', 0), "#e74c3c")
        st.dataframe(pd.DataFrame(d.get('news_evidence', [])), use_container_width=True, hide_index=True)
        with st.container(border=True): st.write(f"⚖️ **AI 판결:** {d.get('ai_reason', 'N/A')}")

# --- [8. UI 레이아웃] ---
st.title("⚖️ Fact-Check Center v97.0")

with st.container(border=True):
    st.markdown("### 🛡️ 법적 고지")
    st.caption("본 서비스는 AI 보조 도구로 최종 판단 책임은 사용자에게 있습니다.")
    agree = st.checkbox("동의함")

url_input = st.text_input("🔗 URL")
if st.button("🚀 분석", disabled=not agree): run_forensic_main(url_input)

st.divider()
st.subheader("🗂️ 분석 히스토리 및 관리")

try:
    resp = supabase.table("analysis_history").select("*").order("id", desc=True).execute()
    df = pd.DataFrame(resp.data)
    if not df.empty:
        if st.session_state["is_admin"]:
            # [관리자 전용: 데이터 삭제 및 편집]
            st.warning("⚠️ 관리자 모드: 데이터 삭제가 가능합니다.")
            df['Select'] = False
            edited_df = st.data_editor(df[['Select', 'id', 'analysis_date', 'video_title', 'fake_prob']], hide_index=True, use_container_width=True)
            if st.button("🗑️ 선택 항목 삭제"):
                to_delete = edited_df[edited_df.Select]
                for _, row in to_delete.iterrows():
                    supabase.table("analysis_history").delete().eq("id", row['id']).execute()
                st.success("삭제 완료"); st.rerun()
        else:
            st.dataframe(df[['analysis_date', 'video_title', 'fake_prob']], hide_index=True, use_container_width=True)
except: pass

# [관리자 전용 센터]
with st.expander("🔐 관리자 전용 통제실"):
    if not st.session_state["is_admin"]:
        if st.text_input("Password", type="password") == ADMIN_PASSWORD:
            st.session_state["is_admin"] = True
            st.rerun()
    else:
        st.success("관리자 인증됨")
        # 1. 모델 상태
        st.write("**🤖 시스템 가용 모델:**")
        st.code(", ".join(get_all_available_models(GOOGLE_API_KEY_A)))
        # 2. 로그 확인
        st.write("**📜 실시간 디버그 로그:**")
        st.text_area("Logs", "\n".join(st.session_state["debug_logs"]), height=200)
        if st.button("로그아웃"):
            st.session_state["is_admin"] = False
            st.rerun()
