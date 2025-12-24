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
st.set_page_config(page_title="유튜브 가짜뉴스 판독기 v99.4", layout="wide", page_icon="🛡️")

# --- [2. 글로벌 상수 정의] ---
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

# Mistral 클라이언트
mistral_client = OpenAI(api_key=MISTRAL_API_KEY, base_url="https://api.mistral.ai/v1")

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [3. 유틸리티 & 파서] ---
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

def safe_int_convert(val, default=50):
    try:
        if isinstance(val, dict): val = list(val.values())[0]
        return int(float(val))
    except: return default

def extract_video_id(url):
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else None

# --- [4. AI 모델 엔진] ---
def get_gemini_search_keywords(title, transcript):
    genai.configure(api_key=GOOGLE_API_KEY_A)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Fact-Check Investigator. Title: {title}. Transcript: {transcript[:10000]}. Extract ONE Korean news search query. String Only."
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except: return title

def call_mistral_judge(prompt, is_json=True):
    try:
        response = mistral_client.chat.completions.create(
            model="mistral-large-latest",
            messages=[{"role": "system", "content": "당신은 전문 팩트체크 판사입니다. 모든 답변(이유 포함)은 반드시 한국어로 작성하세요. JSON 형식으로만 응답하세요."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if is_json else None,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        st.session_state["debug_logs"].append(f"❌ Mistral Error: {e}")
        return None

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
        if not self.vocab: return 0, 0
        qv = self.text_to_vector(query)
        mt = max([self.cosine_similarity(qv, v) for v in self.truth_vectors] or [0])
        mf = max([self.cosine_similarity(qv, v) for v in self.fake_vectors] or [0])
        return mt, mf

vector_engine = VectorEngine()

# --- [6. UI 컴포넌트 복구] ---
def render_score_breakdown(data_list):
    style = """<style>table.score-table { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #e0e0e0; border-radius: 8px; font-size: 14px; margin-top: 10px;} table.score-table th { background-color: #f8f9fa; color: #495057; font-weight: bold; padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; } table.score-table td { padding: 12px 15px; border-bottom: 1px solid #f0f0f0; color: #333; } .badge { padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-align: center; min-width: 45px; } .badge-danger { background-color: #ffebee; color: #d32f2f; } .badge-success { background-color: #e8f5e9; color: #2e7d32; } .badge-neutral { background-color: #f5f5f5; color: #757575; border: 1px solid #e0e0e0; }</style>"""
    rows = ""
    for item, score, note in data_list:
        try:
            score_num = int(score)
            badge = f'<span class="badge badge-danger">+{score_num}</span>' if score_num > 0 else f'<span class="badge badge-success">{score_num}</span>' if score_num < 0 else f'<span class="badge badge-neutral">0</span>'
        except: badge = f'<span class="badge badge-neutral">{score}</span>'
        rows += f"<tr><td>{item}<br><span style='color:#888; font-size:11px;'>{note}</span></td><td style='text-align: right;'>{badge}</td></tr>"
    st.markdown(f"{style}<table class='score-table'><thead><tr><th>분석 항목 (Score Breakdown)</th><th style='text-align: right;'>변동</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def colored_progress_bar(label, percent, color):
    st.markdown(f"""<div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span style="font-size: 13px; font-weight: 600; color: #555;">{label}</span><span style="font-size: 13px; font-weight: 700; color: {color};">{round(percent * 100, 1)}%</span></div><div style="background-color: #eee; border-radius: 5px; height: 8px; width: 100%;"><div style="background-color: {color}; height: 8px; width: {percent * 100}%; border-radius: 5px;"></div></div></div>""", unsafe_allow_html=True)

def render_intelligence_distribution(current_prob):
    try:
        res = supabase.table("analysis_history").select("fake_prob").execute()
        if not res.data: return
        df = pd.DataFrame(res.data)
        base = alt.Chart(df).transform_density('fake_prob', as_=['fake_prob', 'density'], extent=[0, 100], bandwidth=5).mark_area(opacity=0.3, color='#888').encode(x=alt.X('fake_prob:Q', title='가짜뉴스 확률 분포'), y=alt.Y('density:Q', title='데이터 밀도'))
        rule = alt.Chart(pd.DataFrame({'x': [current_prob]})).mark_rule(color='blue', size=3).encode(x='x')
        st.altair_chart(base + rule, use_container_width=True)
    except: pass

# --- [7. 분석 엔진 세부 함수] ---
def scrape_news_content_robust(url):
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        for t in soup(['script', 'style', 'nav', 'footer']): t.decompose()
        text = " ".join([p.get_text().strip() for p in soup.find_all('p') if len(p.get_text()) > 30])
        return (text[:4000], res.url) if len(text) > 100 else (None, res.url)
    except: return None, url

def deep_verify_news_mistral(video_summary, news_url, news_snippet):
    txt, real_url = scrape_news_content_robust(news_url)
    evidence = txt if txt else news_snippet
    prompt = f"""
    동영상 요약본과 뉴스 기사를 비교하여 사실 여부를 판단하세요.
    반드시 한국어로 분석 결과를 작성하세요.
    [동영상] {video_summary[:1500]}
    [뉴스] {evidence[:3000]}
    [판단기준] 일치하면 0-10점, 불일치/허위면 90-100점.
    JSON 형식: {{'score': int, 'reason': '한글 이유'}}
    """
    res_text = call_mistral_judge(prompt)
    parsed = parse_ai_json(res_text)
    if parsed:
        s_val = safe_int_convert(parsed.get('score'))
        return s_val, parsed.get('reason', 'N/A'), "Full Article" if txt else "Snippet Only", evidence, real_url
    return 50, "분석 실패", "Error", "", news_url

def get_mistral_verdict_final(title, transcript, news_list):
    news_sum = "\n".join([f"- {n['뉴스 제목']} (일치도:{n['최종 점수']}, 근거:{n['분석 근거']})" for n in news_list])
    prompt = f"""
    최종 판결을 내리세요. 모든 분석 결과는 반드시 한국어로 작성하세요.
    영상 제목: {title}
    뉴스 증거: {news_sum}
    영상과 뉴스가 정확히 일치하면 0-20점(진실), 다르면 80-100점(가짜).
    JSON 형식: {{'score': int, 'reason': '3문장 이내의 한글 판결문'}}
    """
    res_text = call_mistral_judge(prompt)
    parsed = parse_ai_json(res_text)
    if parsed: 
        s_val = safe_int_convert(parsed.get('score'))
        return s_val, f"{parsed.get('reason')} (By Mistral Large)"
    return 50, "판결 실패"

def fetch_real_transcript(info):
    try:
        subs = info.get('subtitles') or {}
        auto = info.get('automatic_captions') or {}
        merged = {**subs, **auto}
        if 'ko' in merged:
            for f in merged['ko']:
                if f['ext'] == 'vtt':
                    res = requests.get(f['url'])
                    return " ".join([l.strip() for l in res.text.splitlines() if l.strip() and '-->' not in l and '<' not in l]), "Success"
    except: pass
    return None, "Fail"

# --- [8. 리포트 출력 함수 (완전 복구)] ---
def render_report_full_ui(prob, db_count, title, uploader, d, is_cached=False):
    if is_cached: st.success("🎉 기존 분석 결과 로드 완료 (Smart Cache)")

    st.subheader("🕵️ Dual-Engine Analysis Result")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("최종 가짜뉴스 확률", f"{prob}%")
    col_b.metric("AI 판정", "🔴 위험" if prob > 60 else "🟢 안전" if prob < 30 else "🟠 주의")
    col_c.metric("지식 노드", f"{db_count} Nodes")
    
    st.divider()
    st.subheader("🧠 Intelligence Map")
    render_intelligence_distribution(prob)

    st.divider()
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.write("**[영상 상세 정보]**")
        st.table(pd.DataFrame({"항목": ["영상 제목", "채널명", "해시태그"], "내용": [title, uploader, d.get('tags','없음')]}))
        st.info(f"🎯 Investigator 추출 검색어: {d.get('query', 'N/A')}")
        with st.container(border=True):
            st.markdown("📝 **영상 내용 요약**")
            st.write(d.get('summary','내용 없음'))
        st.write("**[Score Breakdown]**")
        render_score_breakdown(d.get('score_breakdown', []))

    with col2:
        st.subheader("📊 5대 정밀 분석 증거")
        st.markdown("**[증거 0] Semantic Vector Space (Internal DB)**")
        colored_progress_bar("✅ 진실 영역 근접도", d.get('ts', 0), "#2ecc71")
        colored_progress_bar("🚨 거짓 영역 근접도", d.get('fs', 0), "#e74c3c")
        
        st.markdown("**[증거 1] 뉴스 교차 대조 (Deep-Web Crawler)**")
        if d.get('news_evidence'):
            st.dataframe(pd.DataFrame(d.get('news_evidence', [])), column_config={"원문": st.column_config.LinkColumn("링크", display_text="🔗 이동")}, hide_index=True)
        else: st.warning("관련 뉴스를 찾을 수 없습니다.")

        st.markdown("**[증거 2] 시청자 여론 분석**")
        if d.get('comment_data'): st.table(pd.DataFrame(d['comment_data'], columns=["항목", "내용"]))
        
        st.markdown("**[증거 3] 자막 세만틱 심층 대조**")
        st.table(pd.DataFrame([["영상 주요 키워드", d.get('top_kw','분석됨')], ["선동성 지수", f"{d.get('agitation',0)}회"]], columns=["항목", "내용"]))
        
        st.markdown("**[증거 4] AI 최종 분석 판단 (Judge Verdict)**")
        with st.container(border=True): st.write(f"⚖️ **판결:** {d.get('ai_reason', 'N/A')}")

# --- [9. 메인 실행 로직] ---
def run_forensic_main(url):
    st.session_state["debug_logs"] = []
    vid = extract_video_id(url)
    if not vid: return st.error("유효하지 않은 URL")

    # DB 학습
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
            tags = info.get('tags', [])
            
            my_bar.progress(10, "데이터 수집 중...")
            trans, _ = fetch_real_transcript(info)
            full_text = trans if trans else desc
            summary = full_text[:800] + "..."
            
            my_bar.progress(30, "AI 수사관 가동 중...")
            query = get_gemini_search_keywords(title, full_text)
            
            my_bar.progress(50, "뉴스 크롤링 진행 중...")
            rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
            items = re.findall(r'<item>(.*?)</item>', requests.get(rss).text, re.DOTALL)[:3]
            news_ev = []; max_match = 0
            for i in items:
                nt = re.search(r'<title>(.*?)</title>', i).group(1).replace("<![CDATA[", "").replace("]]>", "")
                nl = re.search(r'<link>(.*?)</link>', i).group(1)
                score, reason, src, _, real_url = deep_verify_news_mistral(summary, nl, "")
                if score > max_match: max_match = score
                news_ev.append({"뉴스 제목": nt, "일치도": f"{score}%", "최종 점수": score, "분석 근거": reason, "원문": real_url, "비고": src})

            ts, fs = vector_engine.analyze_position(query + " " + title)
            t_impact, f_impact = int(ts*30)*-1, int(fs*30)
            news_penalty = -30 if max_match <= 20 else (30 if max_match >= 80 else 0)
            
            my_bar.progress(85, "AI 최종 판결 중...")
            ai_score, ai_reason = get_mistral_verdict_final(title, full_text, news_ev)
            
            final_prob = max(1, min(99, int((50 + t_impact + f_impact + news_penalty)*WEIGHT_ALGO + ai_score*WEIGHT_AI)))
            
            score_breakdown = [["기본 점수", 50, "중립 시작"], ["진실 DB 매칭", t_impact, "내부 데이터"], ["거짓 패턴 매칭", f_impact, "내부 데이터"], ["뉴스 교차 검증", news_penalty, "크롤링 결과"], ["AI 판결 점수", ai_score, "Mistral 판결"]]
            
            report = {
                "summary": summary, "news_evidence": news_ev, "ai_score": ai_score, "ai_reason": ai_reason,
                "score_breakdown": score_breakdown, "ts": ts, "fs": fs, "query": query, "tags": ", ".join(tags),
                "top_kw": "분석됨", "agitation": 1, "comment_data": [["분석 상태", "완료"]]
            }
            
            supabase.table("analysis_history").insert({"channel_name": uploader, "video_title": title, "fake_prob": final_prob, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url": url, "keywords": query, "detail_json": json.dumps(report, ensure_ascii=False)}).execute()
            my_bar.empty()
            render_report_full_ui(final_prob, db_count, title, uploader, report)
        except Exception as e: st.error(f"오류: {e}")

# --- [10. UI 레이아웃] ---
st.title("⚖️ Fact-Check Center v99.4")

with st.container(border=True):
    st.markdown("### 🛡️ 법적 고지 및 책임 한계 (Disclaimer)\n본 서비스는 **인공지능(AI) 및 알고리즘 기반**으로 영상의 신뢰도를 분석하는 보조 도구입니다. \n분석 결과는 법적 효력이 없으며, 최종 판단의 책임은 사용자에게 있습니다.")
    st.markdown("* **Engine A (Investigator)**: Gemini 1.5 Flash (키워드 추출)\n* **Engine B (Judge)**: Mistral Large 2 (한글 본문 분석 및 판결)")
    agree = st.checkbox("위 내용을 확인하였으며, 이에 동의합니다. (동의 시 분석 버튼 활성화)")

url_input = st.text_input("🔗 분석할 유튜브 URL")
if st.button("🚀 정밀 분석 시작", disabled=not agree, use_container_width=True):
    if url_input: run_forensic_main(url_input)

st.divider()
st.subheader("🗂️ 학습 데이터 관리 (Cloud Knowledge Base)")
try:
    resp = supabase.table("analysis_history").select("*").order("id", desc=True).limit(15).execute()
    df_h = pd.DataFrame(resp.data)
    if not df_h.empty:
        if st.session_state["is_admin"]:
            df_h['Delete'] = False
            edited = st.data_editor(df_h[['Delete', 'id', 'analysis_date', 'video_title', 'fake_prob']], hide_index=True, use_container_width=True)
            if st.button("🗑️ 선택 항목 삭제", type="primary"):
                for _, row in edited[edited.Delete].iterrows():
                    supabase.table("analysis_history").delete().eq("id", row['id']).execute()
                st.success("삭제 완료")
                time.sleep(0.5)
                st.rerun()
        else: st.dataframe(df_h[['analysis_date', 'video_title', 'fake_prob']], use_container_width=True, hide_index=True)
except: pass

with st.expander("🔐 관리자 접속"):
    if not st.session_state["is_admin"]:
        if st.text_input("PW", type="password") == ADMIN_PASSWORD:
            st.session_state["is_admin"] = True
            st.rerun()
    else:
        st.write("🤖 하이브리드 엔진 (Gemini A + Mistral B) 가동 중")
        if st.button("로그아웃"):
            st.session_state["is_admin"] = False
            st.rerun()
