import streamlit as st
from supabase import create_client, Client
import re
import requests
import time
import random
import math
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI  # Mistral 호출용 (OpenAI 호환 규격)
from datetime import datetime
from collections import Counter
import yt_dlp
import pandas as pd
import altair as alt
import json
from bs4 import BeautifulSoup

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="유튜브 가짜뉴스 판독기 v99", layout="wide", page_icon="⚖️")

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
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"] # Mistral 키 추가
except Exception as e:
    st.error(f"❌ 필수 키 설정 누락: {e}")
    st.stop()

# Mistral 클라이언트 초기화
mistral_client = OpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1" # Mistral 공식 엔드포인트
)

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [2. 유틸리티: JSON 파서] ---
def parse_ai_json(text):
    try:
        parsed = json.loads(text)
    except:
        try:
            text = re.sub(r'```json\s*', '', text).replace('```', '')
            match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if match: parsed = json.loads(match.group(1))
            else: return None
        except: return None
    if isinstance(parsed, list):
        return parsed[0] if len(parsed) > 0 and isinstance(parsed[0], dict) else None
    return parsed if isinstance(parsed, dict) else None

def extract_video_id(url):
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else None

# --- [3. 모델 엔진 분리] ---

# [Engine A] Gemini (Investigator)
def get_gemini_search_keywords(title, transcript):
    genai.configure(api_key=GOOGLE_API_KEY_A)
    # 가용한 Gemini 모델 중 하나 선택
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"Fact-Check Investigator. Title: {title}. Transcript: {transcript[:10000]}. Extract ONE Korean news search query (Proper Noun + Core Issue). String Only."
    
    try:
        response = model.generate_content(prompt)
        st.session_state["debug_logs"].append(f"✅ Key A (Gemini) Success")
        return response.text.strip()
    except Exception as e:
        st.session_state["debug_logs"].append(f"❌ Key A (Gemini) Failed: {e}")
        return title

# [Engine B] Mistral Large (Judge) - 신규 도입
def call_mistral_judge(prompt, is_json=True):
    try:
        response = mistral_client.chat.completions.create(
            model="mistral-large-latest", # Mistral의 최고 성능 모델
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if is_json else None,
            temperature=0.1 # 일관된 판단을 위해 온도를 낮춤
        )
        content = response.choices[0].message.content
        st.session_state["debug_logs"].append(f"✅ Key B (Mistral) Success")
        return content
    except Exception as e:
        st.session_state["debug_logs"].append(f"❌ Key B (Mistral) Failed: {e}")
        return None

# --- [4. VectorEngine] ---
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

# --- [5. 팩트체크 세부 로직] ---

def scrape_news_content_robust(url):
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5, allow_redirects=True)
        soup = BeautifulSoup(res.text, 'html.parser')
        for t in soup(['script', 'style', 'nav', 'footer', 'header']): t.decompose()
        text = " ".join([p.get_text().strip() for p in soup.find_all('p') if len(p.get_text()) > 30])
        return (text[:4000], res.url) if len(text) > 100 else (None, res.url)
    except: return None, url

# [Judge] 뉴스 개별 검증 (Mistral 사용)
def deep_verify_news_mistral(video_summary, news_url, news_snippet):
    txt, real_url = scrape_news_content_robust(news_url)
    evidence = txt if txt else news_snippet
    
    prompt = f"""
    [Task] Compare Video vs News. Determine if the news confirms the video claim.
    [Logic] Match=Truth(Score 0-10), Mismatch=Fake(Score 90-100).
    [Video Context] {video_summary[:1500]}
    [News Article] {evidence[:3000]}
    [Output JSON Format] {{ "score": int, "reason": "korean_reason" }}
    """
    res_text = call_mistral_judge(prompt)
    parsed = parse_ai_json(res_text)
    
    if parsed:
        source_type = "Full Article" if txt else "Snippet Only"
        return parsed.get('score', 50), parsed.get('reason', 'N/A'), source_type, evidence, real_url
    return 50, "Mistral Error", "Error", "", news_url

# [Judge] 최종 판결 (Mistral 사용)
def get_mistral_verdict_final(title, transcript, news_list):
    news_sum = "\n".join([f"- {n['뉴스 제목']} (Score:{n['최종 점수']}, Reason:{n['분석 근거']})" for n in news_list])
    
    prompt = f"""
    [Role] Professional Fact-Check AI Judge.
    [Objective] Final verdict on Video Title: '{title}'.
    [Evidence Provided]
    {news_sum}
    [Logic] If News matches Video accurately -> Score 0-20 (Truth). If News contradicts or Video lies -> Score 80-100 (Fake).
    [Output JSON Format] {{ "score": int, "reason": "3 sentences reasoning in KOREAN" }}
    """
    res_text = call_mistral_judge(prompt)
    parsed = parse_ai_json(res_text)
    
    if parsed:
        return parsed.get('score', 50), f"{parsed.get('reason')} (By Mistral Large)"
    return 50, "Final Judgment Failed"

# --- [6. UI 및 보조 함수] ---

def render_score_breakdown(data_list):
    style = """<style>table.score-table { width: 100%; border-collapse: separate; border: 1px solid #e0e0e0; border-radius: 8px; font-size: 14px; margin-top: 10px;} table.score-table th { background-color: #f8f9fa; padding: 12px; text-align: left; } table.score-table td { padding: 12px; border-bottom: 1px solid #f0f0f0; } .badge-danger { background-color: #ffebee; color: #d32f2f; padding: 4px 8px; border-radius: 4px; font-weight: bold; } .badge-success { background-color: #e8f5e9; color: #2e7d32; padding: 4px 8px; border-radius: 4px; font-weight: bold; }</style>"""
    rows = ""
    for item, score, note in data_list:
        try:
            score_num = int(score)
            badge = f'<span class="badge-danger">+{score_num}</span>' if score_num > 0 else f'<span class="badge-success">{score_num}</span>' if score_num < 0 else "0"
        except: badge = str(score)
        rows += f"<tr><td>{item}<br><small style='color:#888;'>{note}</small></td><td style='text-align:right;'>{badge}</td></tr>"
    st.markdown(f"{style}<table class='score-table'><thead><tr><th>분석 항목</th><th style='text-align:right;'>변동</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def colored_progress_bar(label, percent, color):
    st.markdown(f"""<div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between;"><span style="font-size: 13px; font-weight: 600;">{label}</span><span>{round(percent * 100, 1)}%</span></div><div style="background-color: #eee; height: 8px; border-radius: 5px;"><div style="background-color: {color}; height: 8px; width: {percent * 100}%; border-radius: 5px;"></div></div></div>""", unsafe_allow_html=True)

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

# --- [7. 메인 실행 함수] ---

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
            render_report_full_ui(c['fake_prob'], db_count, c['video_title'], c['channel_name'], d, is_cached=True)
            return
        except: pass

    my_bar = st.progress(0, text="분석 프로세스 가동 중...")
    with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title, uploader, desc = info.get('title',''), info.get('uploader',''), info.get('description','')
            tags = info.get('tags', [])
            
            # 자막 수집
            my_bar.progress(15, text="1단계: 영상 자막 수집 중...")
            trans, _ = fetch_real_transcript(info)
            full_text = trans if trans else desc
            summary = full_text[:800] + "..."

            # Key A (Gemini)
            my_bar.progress(35, text="2단계: AI 수사관(Gemini) 키워드 추출 중...")
            query = get_gemini_search_keywords(title, full_text)

            # 뉴스 크롤링 & Key B (Mistral)
            my_bar.progress(55, text="3단계: 뉴스 교차 대조(Mistral Large) 진행 중...")
            rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
            items = re.findall(r'<item>(.*?)</item>', requests.get(rss).text, re.DOTALL)[:3]
            
            news_ev = []; max_match = 0
            for i in items:
                nt = re.search(r'<title>(.*?)</title>', i).group(1).replace("<![CDATA[", "").replace("]]>", "")
                nl = re.search(r'<link>(.*?)</link>', i).group(1)
                nd = re.search(r'<description>(.*?)</description>', i).group(1)
                
                # Mistral을 이용한 심층 검증
                sb_score, sb_reason, src, _, real_url = deep_verify_news_mistral(summary, nl, nd)
                if sb_score > max_match: max_match = sb_score
                news_ev.append({"뉴스 제목": nt, "일치도": f"{sb_score}%", "최종 점수": sb_score, "분석 근거": sb_reason, "원문": real_url, "비고": src})

            # 점수 계산 로직
            ts, fs = vector_engine.analyze_position(query + " " + title)
            t_impact, f_impact = int(ts*30)*-1, int(fs*30)
            news_penalty = -30 if max_match <= 20 else (30 if max_match >= 80 else 0)
            
            # 최종 판결 (Mistral)
            my_bar.progress(85, text="4단계: AI 판사(Mistral) 최종 판결 중...")
            ai_score, ai_reason = get_mistral_verdict_final(title, full_text, news_ev)
            
            final_prob = max(1, min(99, int((50 + t_impact + f_impact + news_penalty)*WEIGHT_ALGO + ai_score*WEIGHT_AI)))
            
            score_breakdown = [["기본 점수", 50, "중립 시작"], ["진실 DB 매칭", t_impact, ""], ["거짓 패턴 매칭", f_impact, ""], ["뉴스 교차 검증", news_penalty, ""], ["AI 최종 판결", ai_score, ""]]
            
            report = {
                "summary": summary, "news_evidence": news_ev, "ai_score": ai_score, "ai_reason": ai_reason,
                "score_breakdown": score_breakdown, "ts": ts, "fs": fs, "query": query, "tags": ", ".join(tags)
            }
            
            supabase.table("analysis_history").insert({"channel_name": uploader, "video_title": title, "fake_prob": final_prob, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url": url, "keywords": query, "detail_json": json.dumps(report, ensure_ascii=False)}).execute()
            my_bar.empty()
            render_report_full_ui(final_prob, db_count, title, uploader, report)

        except Exception as e: st.error(f"분석 중 오류: {e}")

def render_report_full_ui(prob, db_count, title, uploader, d, is_cached=False):
    if is_cached: st.success("🎉 기존 분석 결과 로드 (Smart Cache)")

    st.subheader("🕵️ Dual-Engine Analysis Result")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("최종 가짜뉴스 확률", f"{prob}%")
    col_b.metric("AI 판정", "🔴 위험" if prob > 60 else "🟢 안전" if prob < 30 else "🟠 주의")
    col_c.metric("지식 노드", f"{db_count} Nodes")
    
    st.divider()
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.write(f"**제목:** {title}\n**채널:** {uploader}")
        st.info(f"🎯 검색어: {d.get('query', 'N/A')}")
        with st.container(border=True):
            st.markdown("📝 **내용 요약**")
            st.write(d.get('summary','내용 없음'))
        render_score_breakdown(d.get('score_breakdown', []))

    with col2:
        st.write("📊 **5대 정밀 분석 증거**")
        colored_progress_bar("✅ 진실 영역 근접도", d.get('ts', 0), "#2ecc71")
        colored_progress_bar("🚨 거짓 영역 근접도", d.get('fs', 0), "#e74c3c")
        st.markdown("**[증거 1] 뉴스 교차 대조 (Deep Crawling)**")
        st.dataframe(pd.DataFrame(d.get('news_evidence', [])), column_config={"원문": st.column_config.LinkColumn("링크", display_text="🔗 이동")}, hide_index=True)
        with st.container(border=True):
            st.write(f"⚖️ **AI 판결:** {d.get('ai_reason', 'N/A')}")

# --- [8. UI 메인 레이아웃] ---
st.title("⚖️ Fact-Check Center v99.0")

with st.container(border=True):
    st.markdown("### 🛡️ 법적 고지 및 책임 한계 (Disclaimer)")
    st.markdown("본 서비스는 **인공지능(AI) 및 알고리즘 기반**으로 영상의 신뢰도를 분석하는 보조 도구입니다. 최종 판단의 책임은 사용자에게 있습니다.")
    st.markdown("* **Engine A (Investigator)**: Gemini 1.5 Flash (키워드 추출)\n* **Engine B (Judge)**: Mistral Large 2 (본문 분석 및 판결)")
    agree = st.checkbox("위 내용을 확인하였으며, 이에 동의합니다.")

url_input = st.text_input("🔗 유튜브 URL")
if st.button("🚀 정밀 분석 시작", disabled=not agree, use_container_width=True):
    if url_input: run_forensic_main(url_input)

st.divider()
st.subheader("🗂️ 학습 데이터 관리")
try:
    resp = supabase.table("analysis_history").select("*").order("id", desc=True).limit(20).execute()
    df = pd.DataFrame(resp.data)
    if not df.empty:
        if st.session_state["is_admin"]:
            df['Delete'] = False
            edited = st.data_editor(df[['Delete', 'id', 'analysis_date', 'video_title', 'fake_prob']], hide_index=True, use_container_width=True)
            if st.button("🗑️ 선택 삭제"):
                for _, row in edited[edited.Delete].iterrows():
                    supabase.table("analysis_history").delete().eq("id", row['id']).execute()
                st.rerun()
        else:
            st.dataframe(df[['analysis_date', 'video_title', 'fake_prob']], use_container_width=True, hide_index=True)
except: pass

with st.expander("🔐 관리자 접속"):
    if not st.session_state["is_admin"]:
        if st.text_input("PW", type="password") == ADMIN_PASSWORD:
            st.session_state["is_admin"] = True
            st.rerun()
    else:
        st.write(f"**🤖 엔진 상태:** Gemini(A) + Mistral(B) Active")
        if st.session_state["debug_logs"]:
            st.text_area("Debug Logs", "\n".join(st.session_state["debug_logs"]))
        if st.button("로그아웃"):
            st.session_state["is_admin"] = False
            st.rerun()
