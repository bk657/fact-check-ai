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

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False
if "debug_logs" not in st.session_state:
    st.session_state["debug_logs"] = []

# 🌟 Secrets 로드 (3중 키 로드)
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    GOOGLE_API_KEY_A = st.secrets["GOOGLE_API_KEY_A"]
    GOOGLE_API_KEY_B = st.secrets["GOOGLE_API_KEY_B"]
except:
    st.error("❌ secrets.toml 파일에 API Key 설정이 필요합니다. (Mistral, Google A, Google B)")
    st.stop()

@st.cache_resource
def init_clients():
    # Supabase & Mistral (Gemini는 호출 시마다 키 변경)
    su = create_client(SUPABASE_URL, SUPABASE_KEY)
    mi = Mistral(api_key=MISTRAL_API_KEY)
    return su, mi

supabase, mistral_client = init_clients()

# --- [2. 모델 정의] ---
# Mistral 우선순위 리스트
MISTRAL_MODELS = [
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "open-mixtral-8x22b"
]

# Gemini 모델 탐색 함수 (키 별로 동작)
def get_gemini_models_dynamic(api_key):
    """특정 API Key로 사용 가능한 모델 리스트를 가져옴"""
    genai.configure(api_key=api_key)
    try:
        models = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        models.sort(key=lambda x: 0 if 'flash' in x else 1 if 'pro' in x else 2)
        return models
    except:
        return ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"] # 실패 시 기본값

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

# --- [4. ⭐ Triple Hybrid Survivor Logic] ---
def call_triple_survivor(prompt, is_json=False):
    logs = []
    
    # === [Phase 1: Mistral AI (1선발)] ===
    response_format = {"type": "json_object"} if is_json else None
    for model_name in MISTRAL_MODELS:
        try:
            resp = mistral_client.chat.complete(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                temperature=0.2
            )
            if resp.choices:
                content = resp.choices[0].message.content
                logs.append(f"✅ Success (Mistral): {model_name}")
                return content, f"{model_name}", logs
        except Exception as e:
            logs.append(f"❌ Mistral Failed ({model_name}): {str(e)[:30]}...")
            time.sleep(0.2)
            continue

    # === [Phase 2: Google Gemini Key A (2선발)] ===
    logs.append("⚠️ Mistral 전멸 -> Gemini Key A 투입")
    models_a = get_gemini_models_dynamic(GOOGLE_API_KEY_A)
    
    generation_config = {"response_mime_type": "application/json"} if is_json else {}
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # 키 A 설정 (중요: 재설정)
    genai.configure(api_key=GOOGLE_API_KEY_A)
    
    for model_name in models_a:
        try:
            model = genai.GenerativeModel(model_name, generation_config=generation_config)
            resp = model.generate_content(prompt, safety_settings=safety_settings)
            if resp.text:
                logs.append(f"✅ Success (Gemini Key A): {model_name}")
                return resp.text, f"{model_name} (Key A)", logs
        except Exception as e:
            continue

    # === [Phase 3: Google Gemini Key B (최후의 보루)] ===
    logs.append("⚠️ Key A 전멸 -> Gemini Key B 투입 (Final Stand)")
    
    # 키 B 설정 (중요: 재설정)
    genai.configure(api_key=GOOGLE_API_KEY_B)
    models_b = get_gemini_models_dynamic(GOOGLE_API_KEY_B) # 모델 리스트 다시 확보
    
    for model_name in models_b:
        try:
            model = genai.GenerativeModel(model_name, generation_config=generation_config)
            resp = model.generate_content(prompt, safety_settings=safety_settings)
            if resp.text:
                logs.append(f"✅ Success (Gemini Key B): {model_name}")
                return resp.text, f"{model_name} (Key B)", logs
        except Exception as e:
            continue

    return None, "All Failed (Mistral + Key A + Key B)", logs

# --- [5. 상수 및 데이터] ---
WEIGHT_ALGO = 0.85
WEIGHT_AI = 0.15

OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']
CRITICAL_STATE_KEYWORDS = ['별거', '이혼', '파경', '사망', '위독', '구속', '체포', '실형', '불화', '폭로', '충격', '논란', '중태', '심정지', '뇌사', '압수수색', '소환', '파산', '빚더미', '전과', '감옥', '간첩']

STATIC_TRUTH_CORPUS = ["박나래 위장전입 무혐의", "임영웅 암표 대응", "정희원 저속노화", "대전 충남 통합", "선거 출마 선언"]
STATIC_FAKE_CORPUS = ["충격 폭로 경악", "긴급 속보 소름", "충격 발언 논란", "구속 영장 발부", "영상 유출", "계시 예언", "사형 집행", "위독설"]

class VectorEngine:
    def __init__(self):
        self.vocab = set(); self.truth_vectors = []; self.fake_vectors = []
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

# [Engine A] 수사관
def get_hybrid_search_keywords(title, transcript):
    context_data = transcript[:15000] 
    prompt = f"""
    You are a Fact-Check Investigator.
    [Input] Title: {title}, Transcript: {context_data}
    [Task] Extract ONE precise Google News search query.
    [Rules] Focus on Proper Nouns (Person, Drug, Event). Ignore Generic Verbs.
    [Output] ONLY the Korean search query string (2-4 words). Do not add quotes.
    """
    result_text, model_used, logs = call_triple_survivor(prompt)
    st.session_state["debug_logs"].extend([f"[Key A] {l}" for l in logs])
    return (result_text.strip(), f"✨ {model_used}") if result_text else (title, "❌ Error")

# [크롤러] 뉴스 본문 수집
def scrape_news_content_robust(google_url):
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        response = session.get(google_url, timeout=5, allow_redirects=True)
        final_url = response.url
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'iframe']): tag.decompose()
        text = " ".join([p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 30])
        return (text[:4000], final_url) if len(text) > 100 else (None, final_url)
    except: return None, google_url

# [Engine B] 뉴스 정밀 대조
def deep_verify_news(video_summary, news_url, news_snippet):
    scraped_text, real_url = scrape_news_content_robust(news_url)
    evidence_text = scraped_text if scraped_text else news_snippet
    source_type = "Full Article" if scraped_text else "Snippet Only"
    
    prompt = f"""
    Compare Video Summary vs News Evidence.
    [Video] {video_summary[:2000]}
    [News ({source_type})] {evidence_text}
    [Task] Does news confirm video claim? Match(90-100), Related(40-60), Mismatch(0-10).
    [Output JSON] {{ "score": <int>, "reason": "<short korean reason>" }}
    """
    result_text, model_used, logs = call_triple_survivor(prompt, is_json=True)
    st.session_state["debug_logs"].extend([f"[Verify] {l}" for l in logs])
    
    res = parse_llm_json(result_text)
    if res: return res.get('score', 0), res.get('reason', 'N/A'), source_type, evidence_text, real_url
    return 0, "Error", "Error", "", news_url

# [Engine B] 최종 판결
def get_hybrid_verdict_final(title, transcript, verified_news_list):
    news_summary = ""
    for item in verified_news_list:
        news_summary += f"- News: {item['뉴스 제목']} (Score: {item['최종 점수']}, Reason: {item['분석 근거']})\n"
    
    full_context = transcript[:30000]
    prompt = f"""
    You are a Fact-Check Judge.
    [Video] {title} / {full_context[:2000]}...
    [Evidence] {news_summary}
    [Instruction] Verify truth. Match->Truth(0-30), Mismatch->Fake(70-100). 
    Output JSON format only: {{ "score": <int>, "reason": "<korean explanation>" }}
    """
    result_text, model_used, logs = call_triple_survivor(prompt, is_json=True)
    st.session_state["debug_logs"].extend([f"[Judge] {l}" for l in logs])
    
    res = parse_llm_json(result_text)
    if res: return res.get('score', 50), f"{res.get('reason')} (By {model_used})"
    return 50, "Judge Failed"

# --- [B2B 리포트 생성 엔진 (수정 완료)] ---
def generate_b2b_report_logic(df_history):
    if df_history.empty: return pd.DataFrame()
    
    # 1. 데이터 강제 형변환 (NaN은 0으로) -> 'avg_risk' 에러 해결의 핵심
    df_history['fake_prob'] = pd.to_numeric(df_history['fake_prob'], errors='coerce').fillna(0)
    
    # 2. 안전한 GroupBy (직접 계산 방식)
    grouped = df_history.groupby('channel_name')
    
    # 3. 컬럼별 독립 계산 후 병합 (오류 가능성 차단)
    report = pd.DataFrame({
        'analyzed_count': grouped['fake_prob'].count(),
        'avg_risk': grouped['fake_prob'].mean(),
        'max_risk': grouped['fake_prob'].max(),
        'all_keywords': grouped['keywords'].apply(lambda x: ' '.join([str(k) for k in x if k]))
    }).reset_index()
    
    results = []
    for _, row in report.iterrows():
        avg_score = row['avg_risk']
        
        if avg_score >= 60: grade = "⛔ BLACKLIST (심각)"
        elif avg_score >= 40: grade = "⚠️ CAUTION (주의)"
        else: grade = "✅ SAFE (양호)"
        
        tokens = re.findall(r'[가-힣]{2,}', str(row['all_keywords']))
        targets = ", ".join([t[0] for t in Counter(tokens).most_common(3)])
        
        results.append({
            "채널명": row['channel_name'],
            "위험 등급": grade,
            "평균 가짜 확률": f"{int(avg_score)}%",
            "최고 가짜 확률": f"{int(row['max_risk'])}%",
            "분석 영상 수": f"{int(row['analyzed_count'])}개",
            "주요 타겟": targets
        })
        
    return pd.DataFrame(results).sort_values(by='평균 가짜 확률', ascending=False)

# --- [6. 유틸리티 2] ---
def normalize_korean_word(word):
    word = re.sub(r'[^가-힣0-9]', '', word)
    for j in ['은','는','이','가','을','를','의','에','에게','로','으로']:
        if word.endswith(j): return word[:-len(j)]
    return word

def extract_top_keywords_from_transcript(text, top_n=5):
    raw = re.findall(r'[가-힣]{2,}', text)
    noise = ['충격','속보','긴급','오늘','지금','결국','뉴스','영상']
    tokens = [normalize_korean_word(w) for w in raw if w not in noise]
    return Counter(tokens).most_common(top_n)

def train_dynamic_vector_engine():
    try:
        res_t = supabase.table("analysis_history").select("video_title").lt("fake_prob", 40).execute()
        res_f = supabase.table("analysis_history").select("video_title").gt("fake_prob", 60).execute()
        dt = [r['video_title'] for r in res_t.data] if res_t.data else []
        df = [r['video_title'] for r in res_f.data] if res_f.data else []
        vector_engine.train(STATIC_TRUTH_CORPUS + dt, STATIC_FAKE_CORPUS + df)
        return len(dt)+len(df), dt, df
    except: 
        vector_engine.train(STATIC_TRUTH_CORPUS, STATIC_FAKE_CORPUS)
        return 0, [], []

def save_analysis(channel, title, prob, url, keywords, report_data):
    try: supabase.table("analysis_history").insert({"channel_name": channel, "video_title": title, "fake_prob": prob, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url": url, "keywords": keywords, "detail_json": json.dumps(report_data, ensure_ascii=False)}).execute()
    except: pass

def render_score_breakdown(data_list):
    style = """<style>table.score-table { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; font-family: sans-serif; font-size: 14px; margin-top: 10px;} table.score-table th { background-color: #f8f9fa; color: #495057; font-weight: bold; padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; } table.score-table td { padding: 12px 15px; border-bottom: 1px solid #f0f0f0; color: #333; } table.score-table tr:last-child td { border-bottom: none; } .badge { padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-align: center; min-width: 45px; } .badge-danger { background-color: #ffebee; color: #d32f2f; } .badge-success { background-color: #e8f5e9; color: #2e7d32; } .badge-neutral { background-color: #f5f5f5; color: #757575; border: 1px solid #e0e0e0; }</style>"""
    rows = ""
    for item, score, note in data_list:
        try: val = int(score); badge = f'<span class="badge badge-danger">+{val}</span>' if val > 0 else f'<span class="badge badge-success">{val}</span>' if val < 0 else f'<span class="badge badge-neutral">0</span>'
        except: badge = f'<span class="badge badge-neutral">{score}</span>'
        rows += f"<tr><td>{item}<br><span style='color:#888; font-size:11px;'>{note}</span></td><td style='text-align: right;'>{badge}</td></tr>"
    st.markdown(f"{style}<table class='score-table'><thead><tr><th>분석 항목 (Score Breakdown)</th><th style='text-align: right;'>변동</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def colored_progress_bar(label, percent, color):
    st.markdown(f"""<div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span style="font-size: 13px; font-weight: 600; color: #555;">{label}</span><span style="font-size: 13px; font-weight: 700; color: {color};">{round(percent * 100, 1)}%</span></div><div style="background-color: #eee; border-radius: 5px; height: 8px; width: 100%;"><div style="background-color: {color}; height: 8px; width: {percent * 100}%; border-radius: 5px;"></div></div></div>""", unsafe_allow_html=True)

def summarize_transcript(text, title): return text[:800] + "..."
def clean_html_regex(text): return re.sub('<.*?>', '', text).strip()
def check_is_official(ch): return any(o in ch.upper().replace(" ","") for o in OFFICIAL_CHANNELS)
def count_sensational_words(text): return sum(text.count(w) for w in ['충격', '경악', '실체', '폭로', '속보'])
def detect_ai_content(info): return False, ""

# --- [UI 출력 복구] ---
def render_report_full_ui(prob, db_count, title, uploader, d, is_cached=False):
    if is_cached: st.success("🎉 기존 분석 결과 발견 (Smart Cache)")

    st.subheader(f"🕵️ Triple-Engine Analysis Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("최종 가짜뉴스 확률", f"{prob}%", delta=f"AI Judge Score")
    verdict = "안전 (Verified)" if prob < 30 else "위험 (Fake/Bias)" if prob > 60 else "주의 (Caution)"
    c2.metric("종합 AI 판정", f"{verdict}")
    c3.metric("AI Intelligence Level", f"{db_count} Nodes", delta="Triple Active")
    
    st.divider()
    render_intelligence_distribution(prob)
    
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.write("**[영상 상세 정보]**")
        st.table(pd.DataFrame({"항목": ["영상 제목", "채널명", "해시태그"], "내용": [title, uploader, d.get('tags','없음')]}))
        st.info(f"🎯 **Investigator 추출 검색어**: {d.get('query', 'N/A')}")
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
        st.write("---")

        st.markdown(f"**[증거 1] 뉴스 교차 대조 (Deep-Web Crawler)**")
        if d.get('news_evidence'):
            st.dataframe(pd.DataFrame(d.get('news_evidence', [])), column_config={"원문": st.column_config.LinkColumn(label="링크", display_text="🔗 이동")}, hide_index=True)
        else: st.warning("🔍 관련 뉴스를 찾을 수 없습니다. (Silent Echo Risk)")
            
        st.markdown("**[증거 2] 시청자 여론 심층 분석**")
        st.table(pd.DataFrame([
            ["분석 댓글 수", f"{d.get('cmt_count',0)}개"],
            ["최다 빈출 키워드", ", ".join(d.get('top_cmt_kw', []))],
            ["논란 감지 건수", f"{d.get('red_cnt',0)}회"],
            ["주제 일치도", d.get('cmt_rel', '')]
        ], columns=["항목", "내용"]))
        
        st.markdown("**[증거 3] 자막 세만틱 심층 대조**")
        st.table(pd.DataFrame([
            ["영상 주요 키워드", "분석 완료"],
            ["선동성 지수", f"{d.get('agitation',0)}회"]
        ], columns=["분석 항목", "판정 결과"]))
        
        st.markdown("**[증거 4] AI 최종 분석 판단 (Judge Verdict)**")
        with st.container(border=True):
            st.write(f"⚖️ **판결:** {d.get('ai_reason', 'N/A')}")
            st.caption(f"* Triple 독립 추론 점수: {d.get('ai_score', 0)}점")

def run_forensic_main(url):
    st.session_state["debug_logs"] = []
    my_bar = st.progress(0, text="Triple Engine 가동 중...")
    db_count, _, _ = train_dynamic_vector_engine()
    
    vid = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if vid: vid = vid.group(1)

    # 캐시 체크
    cached_res = supabase.table("analysis_history").select("*").ilike("video_url", f"%{vid}%").order("id", desc=True).limit(1).execute()
    if cached_res.data:
        c = cached_res.data[0]
        try:
            d = json.loads(c.get('detail_json', '{}'))
            render_report_full_ui(c['fake_prob'], db_count, c['video_title'], c['channel_name'], d, is_cached=True)
            return
        except: pass

    with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', ''); uploader = info.get('uploader', '')
            tags = info.get('tags', []); desc = info.get('description', '')
            
            my_bar.progress(10, "1단계: 데이터 수집...")
            trans, _ = fetch_real_transcript(info)
            full_text = trans if trans else desc
            summary = summarize_transcript(full_text, title)
            
            my_bar.progress(30, "2단계: AI 수사관...")
            query, _ = get_hybrid_search_keywords(title, full_text)

            my_bar.progress(50, "3단계: 뉴스 대조...")
            news_items = fetch_news_regex(query)
            news_ev = []; max_match = 0
            ts, fs = vector_engine.analyze_position(query + " " + title)
            t_impact = int(ts * 30) * -1; f_impact = int(fs * 30)

            for idx, item in enumerate(news_items[:3]):
                ai_s, ai_r, src, txt, real_url = deep_verify_news(summary, item['link'], item['desc'])
                if ai_s > max_match: max_match = ai_s
                status_icon = "🟢" if ai_s >= 80 else "🟡" if ai_s >= 60 else "🔴"
                news_ev.append({
                    "뉴스 제목": item['title'],
                    "일치도": f"{status_icon} {ai_s}%",
                    "최종 점수": f"{ai_s}%",
                    "분석 근거": ai_r,
                    "비고": f"[{src}]",
                    "원문": real_url
                })
            
            news_score = -40 if max_match >= 80 else (-15 if max_match >= 70 else (10 if max_match >= 60 else 30)) if news_ev else 0
            
            cmts = fetch_comments_via_api(vid)
            top_cmt, rel_score, rel_msg = analyze_comment_relevance(cmts, title + full_text)
            red_cnt, _ = check_red_flags(cmts)
            
            silent_penalty = 0; is_silent = (len(news_ev) == 0)
            if is_silent:
                if any(k in title for k in CRITICAL_STATE_KEYWORDS): silent_penalty = 10
                elif count_sensational_words(title) >= 3: silent_penalty = 20
            if check_is_official(uploader): news_score = -50; silent_penalty = 0
            
            bait = 10 if any(w in title for w in ['충격','경악','폭로']) else -5
            algo_base = 50 + t_impact + f_impact + news_score + (min(20, red_cnt*3)) + bait + silent_penalty
            
            my_bar.progress(90, "5단계: 최종 판결...")
            ai_judge_score, ai_judge_reason = get_hybrid_verdict_final(title, full_text, news_ev)
            
            neutralized = False
            if t_impact == 0 and f_impact == 0 and is_silent:
                neutralized = True
                ai_judge_score = int((ai_judge_score + 50) / 2)
                algo_base = int((algo_base + 50) / 2)
            
            final_prob = max(1, min(99, int(algo_base * WEIGHT_ALGO + ai_judge_score * WEIGHT_AI)))
            
            score_bd = [
                ["🏁 기본 중립 점수 (Base Score)", 50, "모든 분석은 50점(중립)에서 시작"],
                ["진실 데이터 맥락", t_impact, "내부 DB 진실 데이터와 유사성"],
                ["가짜 패턴 맥락", f_impact, "내부 DB 가짜 데이터와 유사성"],
                ["뉴스 매칭 상태", news_score, "Deep-Crawler 정밀 대조 결과 (Strict)"],
                ["여론/제목/태그 가감", min(20, red_cnt*3) + bait, ""],
                ["-----------------", "", ""],
                ["⚖️ AI Judge Score (15%)", ai_judge_score, "Triple 종합 추론 (참고용)"]
            ]
            
            report_data = {
                "summary": summary, "news_evidence": news_ev, "ai_score": ai_judge_score, "ai_reason": ai_judge_reason,
                "score_breakdown": score_bd, "ts": ts, "fs": fs, "query": query, "tags": ", ".join(tags),
                "cmt_count": len(cmts), "top_cmt_kw": top_cmt, "red_cnt": red_cnt, "cmt_rel": f"{rel_score}% ({rel_msg})",
                "agitation": count_sensational_words(title)
            }
            
            save_analysis(uploader, title, final_prob, url, query, report_data)
            my_bar.empty()
            render_report_full_ui(final_prob, db_count, title, uploader, report_data, is_cached=False)

        except Exception as e: st.error(f"오류: {e}")

# --- [UI] ---
st.title("⚖️유튜브 가짜뉴스 판독기 (Triple Engine)")
with st.container(border=True):
    st.markdown("### 🛡️ 법적 고지 및 책임 한계 (Disclaimer)\n본 서비스는 **인공지능(AI) 및 알고리즘 기반**으로 영상의 신뢰도를 분석하는 보조 도구입니다. \n분석 결과는 법적 효력이 없으며, 최종 판단의 책임은 사용자에게 있습니다.")
    st.markdown("* **1st Line**: Mistral AI\n* **2nd Line**: Google Gemini Key A\n* **3rd Line**: Google Gemini Key B (Final Backup)")
    agree = st.checkbox("위 내용을 확인하였으며, 이에 동의합니다. (동의 시 분석 버튼 활성화)")

url_input = st.text_input("🔗 분석할 유튜브 URL")
if st.button("🚀 정밀 분석 시작", disabled=not agree, use_container_width=True):
    if url_input: run_forensic_main(url_input)

st.divider()
try:
    resp = supabase.table("analysis_history").select("*").order("id", desc=True).execute()
    df = pd.DataFrame(resp.data)
except: df = pd.DataFrame()

if not df.empty and st.session_state["is_admin"]:
    edited = st.data_editor(df, hide_index=True)

# [관리자]
with st.expander("🔐 관리자 접속 (Admin Access)"):
    if st.session_state["is_admin"]:
        st.success("관리자 권한 활성화됨")
        st.divider()
        st.subheader("🏢 B2B 브랜드 세이프티 리포트 (Business Intelligence)")
        if st.button("📊 리포트 생성 및 분석"):
            try:
                # [B2B 리포트 생성 로직 호출]
                rpt = generate_b2b_report_logic(df)
                st.dataframe(
                    rpt,
                    column_config={
                        "위험 등급": st.column_config.TextColumn("Risk Level", help="평균 가짜뉴스 확률 기반 등급"),
                        "평균 가짜 확률": st.column_config.ProgressColumn("Avg Risk", format="%s", min_value=0, max_value=100),
                    },
                    use_container_width=True, hide_index=True
                )
                csv = rpt.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 리포트 엑셀(CSV) 다운로드", csv, "b2b_report.csv", "text/csv")
            except Exception as e: st.error(f"리포트 생성 실패: {e}")
        
        st.divider()
        st.write("**🤖 Triple Defense System Status:**")
        st.caption("1️⃣ Mistral Priority Chain")
        st.code(", ".join(MISTRAL_MODELS))
        
        if st.session_state["debug_logs"]:
            st.write(f"**📜 최근 실행 로그 ({len(st.session_state['debug_logs'])}건):**")
            st.text_area("Logs", "\n".join(st.session_state["debug_logs"]), height=300)
        
        if st.button("로그아웃"): st.session_state["is_admin"]=False; st.rerun()
    else:
        if st.text_input("Admin Password", type="password") == ADMIN_PASSWORD: st.session_state["is_admin"]=True; st.rerun()
