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
st.set_page_config(page_title="Fact-Check Center v96.0", layout="wide", page_icon="⚖️")

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
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']
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

def get_gemini_search_keywords(title, transcript):
    prompt = f"Role: Investigator. Title: {title}. Transcript: {transcript[:15000]}. Extract ONE Korean search query (Proper Noun + Issue). Output string only."
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
    source = "Full Article" if txt else "Snippet Only"
    prompt = f"Summary: {video_summary[:2000]}. News: {evidence}. Match confirms truth(0-10), Mismatch=Fake(90-100). Output JSON {{'score': int, 'reason': 'kr_reason'}}"
    res, model, logs = call_gemini_survivor(GOOGLE_API_KEY_B, prompt, is_json=True)
    st.session_state["debug_logs"].extend([f"[Key B-Verify] {l}" for l in logs])
    parsed = parse_gemini_json(res)
    if parsed: return parsed.get('score', 50), parsed.get('reason', 'N/A'), source, evidence, real_url
    return 50, "Error", "Error", "", news_url

def get_gemini_verdict_final(title, transcript, news_list):
    news_sum = "\n".join([f"- {n['뉴스 제목']} (Score:{n['최종 점수']}, Reason:{n['분석 근거']})" for n in news_list])
    prompt = f"Judge Final Verdict. Video: {title}. Evidence: {news_sum}. Logic: Match=Truth(0-20), Mismatch=Fake(80-100). Output JSON {{'score': int, 'reason': 'kr_reason'}}"
    res, model, logs = call_gemini_survivor(GOOGLE_API_KEY_B, prompt, is_json=True)
    st.session_state["debug_logs"].extend([f"[Key B-Final] {l}" for l in logs])
    parsed = parse_gemini_json(res)
    if parsed: return parsed.get('score', 50), f"{parsed.get('reason')} (By {model})"
    return 50, "Judge Failed"

# --- [6. UI 컴포넌트 복구] ---
def render_score_breakdown(data_list):
    style = """<style>table.score-table { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; font-family: sans-serif; font-size: 14px; margin-top: 10px;} table.score-table th { background-color: #f8f9fa; color: #495057; font-weight: bold; padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; } table.score-table td { padding: 12px 15px; border-bottom: 1px solid #f0f0f0; color: #333; } table.score-table tr:last-child td { border-bottom: none; } .badge { padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-align: center; min-width: 45px; } .badge-danger { background-color: #ffebee; color: #d32f2f; } .badge-success { background-color: #e8f5e9; color: #2e7d32; } .badge-neutral { background-color: #f5f5f5; color: #757575; border: 1px solid #e0e0e0; }</style>"""
    rows = ""
    for item, score, note in data_list:
        try:
            score_num = int(score)
            if score_num > 0: badge = f'<span class="badge badge-danger">+{score_num} (가짜 의심)</span>'
            elif score_num < 0: badge = f'<span class="badge badge-success">{score_num} (진실 입증)</span>'
            else: badge = f'<span class="badge badge-neutral">0</span>'
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

# --- [7. 메인 팩트체크 로직] ---
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

def fetch_comments_via_api(video_id):
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        res = requests.get(url, params={'part': 'snippet', 'videoId': video_id, 'key': YOUTUBE_API_KEY, 'maxResults': 50})
        items = [i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in res.json().get('items', [])]
        return items, "Success"
    except: return [], "Fail"

def fetch_news_regex(query):
    try:
        rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
        raw = requests.get(rss, timeout=5).text
        items = re.findall(r'<item>(.*?)</item>', raw, re.DOTALL)
        res = []
        for i in items[:5]:
            t = re.search(r'<title>(.*?)</title>', i)
            l = re.search(r'<link>(.*?)</link>', i)
            d = re.search(r'<description>(.*?)</description>', i)
            if t and l:
                res.append({'title': t.group(1).replace("<![CDATA[", "").replace("]]>", ""), 'desc': re.sub('<.*?>', '', d.group(1)) if d else "", 'link': l.group(1).strip()})
        return res
    except: return []

# --- [8. 최종 리포트 출력 함수 (복구됨)] ---
def render_final_report_ui(final_prob, db_count, title, uploader, hashtag_display, query, summary, top_kw, agitation, news_ev, ai_judge_score, ai_judge_reason, score_breakdown, ts, fs, cmts=None, is_cached=False):
    if is_cached:
        st.success(f"🎉 **기존 분석 결과 발견! (Fast-Track)**: 이미 분석된 데이터입니다.")

    st.subheader("🕵️ Dual-Engine Analysis Result")
    col_a, col_b, col_c = st.columns(3)
    with col_a: st.metric("최종 가짜뉴스 확률", f"{final_prob}%", delta=f"AI Judge: {ai_judge_score}pt")
    with col_b:
        icon = "🟢" if final_prob < 30 else "🔴" if final_prob > 60 else "🟠"
        verdict = "안전 (Verified)" if final_prob < 30 else "위험 (Fake/Bias)" if final_prob > 60 else "주의 (Caution)"
        st.metric("종합 AI 판정", f"{icon} {verdict}")
    with col_c: st.metric("AI Intelligence Level", f"{db_count} Nodes", delta="Active")
    
    st.divider()
    st.subheader("🧠 Intelligence Map")
    render_intelligence_distribution(final_prob)

    st.divider()
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.write("**[영상 상세 정보]**")
        st.table(pd.DataFrame({"항목": ["영상 제목", "채널명", "해시태그"], "내용": [title, uploader, hashtag_display]}))
        st.info(f"🎯 **Investigator 추출 검색어**: {query}")
        with st.container(border=True):
            st.markdown("📝 **영상 내용 요약**")
            st.write(summary)
        st.write("**[Score Breakdown]**")
        render_score_breakdown(score_breakdown)

    with col2:
        st.subheader("📊 5대 정밀 분석 증거")
        st.markdown("**[증거 0] Semantic Vector Space (Internal DB)**")
        colored_progress_bar("✅ 진실 영역 근접도", ts, "#2ecc71")
        colored_progress_bar("🚨 거짓 영역 근접도", fs, "#e74c3c")
        
        st.markdown("**[증거 1] 뉴스 교차 대조 (Deep-Web Crawler)**")
        if news_ev:
            st.dataframe(pd.DataFrame(news_ev), column_config={"원문": st.column_config.LinkColumn("링크", display_text="🔗 이동")}, hide_index=True)
            with st.expander("🔍 크롤링된 본문 분석 상태 보기"):
                for n in news_ev: st.caption(f"**{n['뉴스 제목']}**: {n.get('비고', '분석 완료')}")
        else: st.warning("관련 뉴스를 찾을 수 없습니다.")

        st.markdown("**[증거 2] 시청자 여론 심층 분석**")
        if cmts: st.info(f"시청자 댓글 {len(cmts)}개를 분석하여 키워드를 추출했습니다.")
        
        st.markdown("**[증거 3] 자막 세만틱 심층 대조**")
        st.table(pd.DataFrame([["영상 주요 키워드", top_kw], ["선동성 지수", f"{agitation}회"]], columns=["항목", "내용"]))
        
        st.markdown("**[증거 4] AI 최종 분석 판단 (Judge Verdict)**")
        with st.container(border=True):
            st.write(f"⚖️ **판결:** {ai_judge_reason}")

# --- [9. 메인 실행 함수] ---
def run_forensic_main(url):
    st.session_state["debug_logs"] = []
    vid = extract_video_id(url)
    if not vid:
        st.error("유효하지 않은 URL입니다.")
        return

    # DB 학습
    res_t = supabase.table("analysis_history").select("video_title").lt("fake_prob", 40).execute()
    res_f = supabase.table("analysis_history").select("video_title").gt("fake_prob", 60).execute()
    dt = [row['video_title'] for row in res_t.data] if res_t.data else []
    df = [row['video_title'] for row in res_f.data] if res_f.data else []
    vector_engine.train(STATIC_TRUTH_CORPUS + dt, STATIC_FAKE_CORPUS + df)
    db_count = len(dt) + len(df)

    # 캐시 체크
    cached_res = supabase.table("analysis_history").select("*").ilike("video_url", f"%{vid}%").order("id", desc=True).limit(1).execute()
    if cached_res.data:
        cached = cached_res.data[0]
        try:
            d = json.loads(cached.get('detail_json', '{}'))
            render_final_report_ui(cached['fake_prob'], db_count, cached['video_title'], cached['channel_name'], "데이터 복원됨", cached['keywords'], d['summary'], d['top_kw'], d['agitation'], d['news_evidence'], d['ai_score'], d['ai_reason'], d['score_breakdown'], d['ts'], d['fs'], is_cached=True)
            return
        except: pass

    my_bar = st.progress(0, text="분석 시작...")
    with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', ''); uploader = info.get('uploader', '')
            tags = info.get('tags', []); desc = info.get('description', '')
            
            my_bar.progress(20, "데이터 수집 중...")
            trans, _ = fetch_real_transcript(info)
            full_text = trans if trans else desc
            summary = full_text[:800] + "..."
            
            my_bar.progress(40, "AI 수사관 가동...")
            query, _ = get_gemini_search_keywords(title, full_text)
            
            my_bar.progress(60, "뉴스 크롤링 중...")
            news_items = fetch_news_regex(query)
            news_ev = []; max_match = 0
            for item in news_items[:3]:
                score, reason, src, _, real_url = deep_verify_news(summary, item['link'], item['desc'])
                if score > max_match: max_match = score
                news_ev.append({"뉴스 제목": item['title'], "일치도": f"{score}%", "최종 점수": score, "분석 근거": reason, "원문": real_url, "비고": src})
            
            news_penalty = -30 if max_match <= 20 else (30 if max_match >= 80 else 0)
            ts, fs = vector_engine.analyze_position(query + " " + title)
            t_impact = int(ts * 30) * -1; f_impact = int(fs * 30)
            
            my_bar.progress(85, "AI 판사 판결 중...")
            ai_score, ai_reason = get_gemini_verdict_final(title, full_text, news_ev)
            
            final_prob = max(1, min(99, int((50 + t_impact + f_impact + news_penalty) * WEIGHT_ALGO + ai_score * WEIGHT_AI)))
            
            score_breakdown = [["기본 점수", 50, "중립 시작"], ["진실 DB 매칭", t_impact, "내부 데이터"], ["거짓 패턴 매칭", f_impact, "내부 데이터"], ["뉴스 교차 검증", news_penalty, "크롤링 결과"], ["AI 최종 판결", ai_score, ai_reason]]
            
            report = {"summary": summary, "top_kw": "추출됨", "agitation": 1, "news_evidence": news_ev, "ai_score": ai_score, "ai_reason": ai_reason, "score_breakdown": score_breakdown, "ts": ts, "fs": fs}
            
            supabase.table("analysis_history").insert({"channel_name": uploader, "video_title": title, "fake_prob": final_prob, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url": url, "keywords": query, "detail_json": json.dumps(report, ensure_ascii=False)}).execute()
            
            my_bar.empty()
            render_final_report_ui(final_prob, db_count, title, uploader, ", ".join(tags), query, summary, "분석됨", 1, news_ev, ai_score, ai_reason, score_breakdown, ts, fs)

        except Exception as e: st.error(f"오류: {e}")

# --- [10. UI 레이아웃] ---
st.title("⚖️ Fact-Check Center v96.0")

# [복구] 법적 고지 섹션
with st.container(border=True):
    st.markdown("### 🛡️ 법적 고지 및 책임 한계 (Disclaimer)")
    st.markdown("본 서비스는 **인공지능(AI) 및 알고리즘 기반**으로 영상의 신뢰도를 분석하는 보조 도구입니다. 분석 결과는 법적 효력이 없으며, 최종 판단의 책임은 사용자에게 있습니다.")
    st.markdown("* **Engine A (Investigator)**: 정밀 키워드 추출 (Full Context)\n* **Engine B (Judge)**: 뉴스 본문 크롤링 및 정밀 대조 (Deep-Web Crawler)")
    agree = st.checkbox("위 내용을 확인하였으며, 이에 동의합니다. (동의 시 분석 활성화)")

url_input = st.text_input("🔗 분석할 유튜브 URL")
if st.button("🚀 정밀 분석 시작", disabled=not agree, use_container_width=True):
    if url_input: run_forensic_main(url_input)

st.divider()
st.subheader("🗂️ 최근 분석 히스토리")
try:
    h = supabase.table("analysis_history").select("analysis_date, video_title, fake_prob, keywords").order("id", desc=True).limit(10).execute()
    if h.data: st.dataframe(pd.DataFrame(h.data), use_container_width=True, hide_index=True)
except: pass

with st.expander("🔐 관리자 접속"):
    if st.text_input("PW", type="password") == ADMIN_PASSWORD:
        st.session_state["is_admin"] = True
        if st.session_state["debug_logs"]: st.text_area("Logs", "\n".join(st.session_state["debug_logs"]))
