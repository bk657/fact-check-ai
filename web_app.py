import streamlit as st
from supabase import create_client, Client
import re
import requests
import time
import random
import math
import os
# --- [변경됨] Mistral AI 라이브러리 임포트 ---
from mistralai import Mistral
from datetime import datetime
from collections import Counter
import yt_dlp
import pandas as pd
import altair as alt
import json
from bs4 import BeautifulSoup

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="유튜브 가짜뉴스 판독기 (Mistral Edition)", layout="wide", page_icon="🛡️")

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
    # --- [변경됨] Mistral API Key 로드 ---
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
except:
    st.error("❌ 필수 키(API Keys)가 설정되지 않았습니다. .streamlit/secrets.toml에 MISTRAL_API_KEY 등을 확인해주세요.")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def init_mistral():
    return Mistral(api_key=MISTRAL_API_KEY)

supabase = init_supabase()
mistral_client = init_mistral()

# --- [2. 유틸리티: JSON 파싱 헬퍼] ---
def parse_llm_json(text):
    """LLM이 리스트로 주든 마크다운을 섞든 무조건 딕셔너리로 변환"""
    try:
        # 1. 순수 파싱 시도
        parsed = json.loads(text)
    except:
        try:
            # 2. 마크다운 제거 후 파싱 시도
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```', '', text)
            # 중괄호나 대괄호로 시작하는 부분 추출
            match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
            else:
                return None
        except:
            return None

    # 리스트면 첫 번째 요소 추출
    if isinstance(parsed, list):
        if len(parsed) > 0 and isinstance(parsed[0], dict):
            return parsed[0]
        else:
            return None 
            
    # 딕셔너리면 그대로 반환
    if isinstance(parsed, dict):
        return parsed
        
    return None

# --- [3. 모델 자동 탐색기 (Mistral 버전)] ---
# Mistral은 모델 리스트가 비교적 고정적이므로 안정적인 모델들을 우선순위대로 배치합니다.
AVAILABLE_MISTRAL_MODELS = [
    "mistral-large-latest",  # 성능 최우선
    "mistral-medium-latest", # 밸런스
    "mistral-small-latest",  # 속도/비용 최우선
    "open-mixtral-8x22b"     # 백업
]

# --- [4. 상수 정의] ---
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

# --- [6. Mistral Logic (변경됨)] ---
def call_mistral_survivor(prompt, is_json=False):
    logs = []
    
    # JSON 포맷 설정
    response_format = {"type": "json_object"} if is_json else None
    
    for model_name in AVAILABLE_MISTRAL_MODELS:
        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            chat_response = mistral_client.chat.complete(
                model=model_name,
                messages=messages,
                response_format=response_format,
                temperature=0.2 # 사실 여부 판단이므로 낮은 온도 설정
            )
            
            if chat_response.choices:
                content = chat_response.choices[0].message.content
                logs.append(f"✅ Success: {model_name}")
                return content, model_name, logs
                
        except Exception as e:
            logs.append(f"❌ Failed ({model_name}): {str(e)[:50]}...")
            time.sleep(0.5)
            continue
            
    return None, "All Failed", logs

# [Engine A] 수사관 (Mistral)
def get_mistral_search_keywords(title, transcript):
    context_data = transcript[:15000] 
    prompt = f"""
    You are a Fact-Check Investigator.
    [Input] Title: {title}, Transcript: {context_data}
    [Task] Extract ONE precise Google News search query.
    [Rules] Focus on Proper Nouns (Person, Drug, Event). Ignore Generic Verbs.
    [Output] ONLY the Korean search query string (2-4 words). Do not add quotes or explanations.
    """
    result_text, model_used, logs = call_mistral_survivor(prompt)
    st.session_state["debug_logs"].extend([f"[Mistral A] {l}" for l in logs])
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

# [Engine B] 뉴스 정밀 대조 (Mistral)
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
    result_text, model_used, logs = call_mistral_survivor(prompt, is_json=True)
    st.session_state["debug_logs"].extend([f"[Mistral B-Verify] {l}" for l in logs])
    
    res = parse_llm_json(result_text)
    if res: return res.get('score', 0), res.get('reason', 'N/A'), source_type, evidence_text, real_url
    return 0, "Error", "Error", "", news_url

# [Engine B] 최종 판결 (Mistral)
def get_mistral_verdict_final(title, transcript, verified_news_list):
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
    result_text, model_used, logs = call_mistral_survivor(prompt, is_json=True)
    st.session_state["debug_logs"].extend([f"[Mistral B-Final] {l}" for l in logs])
    
    res = parse_llm_json(result_text)
    if res: return res.get('score', 50), f"{res.get('reason')} (By {model_used})"
    return 50, "Judge Failed"

# --- [7. 유틸리티 함수] ---
def normalize_korean_word(word):
    word = re.sub(r'[^가-힣0-9]', '', word)
    for j in ['은','는','이','가','을','를','의','에','에게','로','으로']:
        if word.endswith(j): return word[:-len(j)]
    return word

def extract_meaningful_tokens(text):
    raw = re.findall(r'[가-힣]{2,}', text)
    noise = ['충격','속보','긴급','오늘','지금','결국','뉴스','영상']
    return [normalize_korean_word(w) for w in raw if w not in noise]

def extract_top_keywords_from_transcript(text, top_n=5):
    if not text: return []
    tokens = extract_meaningful_tokens(text)
    return Counter(tokens).most_common(top_n)

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

def check_db_similarity(query, truth_list, fake_list):
    return vector_engine.analyze_position(query)

def save_analysis(channel, title, prob, url, keywords):
    try: supabase.table("analysis_history").insert({"channel_name": channel, "video_title": title, "fake_prob": prob, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url": url, "keywords": keywords}).execute()
    except: pass

def render_intelligence_distribution(current_prob):
    try:
        res = supabase.table("analysis_history").select("fake_prob").execute()
        if not res.data: return
        df = pd.DataFrame(res.data)
        base = alt.Chart(df).transform_density('fake_prob', as_=['fake_prob', 'density'], extent=[0, 100], bandwidth=5).mark_area(opacity=0.3, color='#888').encode(x=alt.X('fake_prob:Q', title='가짜뉴스 확률 분포'), y=alt.Y('density:Q', title='데이터 밀도'))
        rule = alt.Chart(pd.DataFrame({'x': [current_prob]})).mark_rule(color='blue', size=3).encode(x='x')
        st.altair_chart(base + rule, use_container_width=True)
    except: pass

def colored_progress_bar(label, percent, color):
    st.markdown(f"""<div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span style="font-size: 13px; font-weight: 600; color: #555;">{label}</span><span style="font-size: 13px; font-weight: 700; color: {color};">{round(percent * 100, 1)}%</span></div><div style="background-color: #eee; border-radius: 5px; height: 8px; width: 100%;"><div style="background-color: {color}; height: 8px; width: {percent * 100}%; border-radius: 5px;"></div></div></div>""", unsafe_allow_html=True)

def render_score_breakdown(data_list):
    style = """<style>table.score-table { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; font-family: sans-serif; font-size: 14px; margin-top: 10px;} table.score-table th { background-color: #f8f9fa; color: #495057; font-weight: bold; padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; } table.score-table td { padding: 12px 15px; border-bottom: 1px solid #f0f0f0; color: #333; } table.score-table tr:last-child td { border-bottom: none; } .badge { padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-align: center; min-width: 45px; } .badge-danger { background-color: #ffebee; color: #d32f2f; } .badge-success { background-color: #e8f5e9; color: #2e7d32; } .badge-neutral { background-color: #f5f5f5; color: #757575; border: 1px solid #e0e0e0; }</style>"""
    rows = ""
    for item, score, note in data_list:
        try:
            score_num = int(score)
            badge = f'<span class="badge badge-danger">+{score_num}</span>' if score_num > 0 else f'<span class="badge badge-success">{score_num}</span>' if score_num < 0 else f'<span class="badge badge-neutral">0</span>'
        except: badge = f'<span class="badge badge-neutral">{score}</span>'
        rows += f"<tr><td>{item}<br><span style='color:#888; font-size:11px;'>{note}</span></td><td style='text-align: right;'>{badge}</td></tr>"
    st.markdown(f"{style}<table class='score-table'><thead><tr><th>분석 항목 (Score Breakdown)</th><th style='text-align: right;'>변동</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def summarize_transcript(text, title, max_sentences=3):
    return text[:800] + "..." if len(text) > 800 else text

def clean_html_regex(text):
    return re.sub('<.*?>', '', text).strip()

def detect_ai_content(info):
    is_ai, reasons = False, []
    text = (info.get('title', '') + " " + info.get('description', '') + " " + " ".join(info.get('tags', []))).lower()
    for kw in ['ai', 'artificial intelligence', 'chatgpt', 'deepfake', 'synthetic', '인공지능', '딥페이크']:
        if kw in text: is_ai = True; reasons.append(f"키워드 감지: {kw}"); break
    return is_ai, ", ".join(reasons)

def check_is_official(channel_name):
    norm_name = channel_name.upper().replace(" ", "")
    return any(o in norm_name for o in OFFICIAL_CHANNELS)

def count_sensational_words(text):
    return sum(text.count(w) for w in ['충격', '경악', '실체', '폭로', '난리', '속보', '긴급', '소름', 'ㄷㄷ'])

def check_tag_abuse(title, hashtags, channel_name):
    if check_is_official(channel_name): return 0, "공식 채널 면제"
    if not hashtags: return 0, "해시태그 없음"
    return 0, "양호"

def fetch_real_transcript(info_dict):
    try:
        url = None
        subs = info_dict.get('subtitles') or {}
        auto = info_dict.get('automatic_captions') or {}
        merged = {**subs, **auto}
        if 'ko' in merged:
            for fmt in merged['ko']:
                if fmt['ext'] == 'vtt': url = fmt['url']; break
        if url:
            res = requests.get(url)
            if res.status_code == 200:
                lines = [l.strip() for l in res.text.splitlines() if l.strip() and '-->' not in l and '<' not in l]
                return " ".join(lines), "Success"
    except: pass
    return None, "Fail"

def fetch_comments_via_api(video_id):
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        res = requests.get(url, params={'part': 'snippet', 'videoId': video_id, 'key': YOUTUBE_API_KEY, 'maxResults': 50})
        if res.status_code == 200:
            data = res.json()
            items = []
            for i in data.get('items', []):
                snippet = i.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
                if 'textDisplay' in snippet: items.append(snippet['textDisplay'])
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

def analyze_comment_relevance(comments, context_text):
    if not comments: return [], 0, "분석 불가"
    cn = extract_meaningful_tokens(" ".join(comments))
    top = Counter(cn).most_common(5)
    ctx = set(extract_meaningful_tokens(context_text))
    match = sum(1 for w,c in top if w in ctx)
    score = int(match/len(top)*100) if top else 0
    msg = "✅ 주제 집중" if score >= 60 else "⚠️ 일부 관련" if score >= 20 else "❌ 무관"
    return [f"{w}({c})" for w, c in top], score, msg

def check_red_flags(comments):
    detected = [k for c in comments for k in ['가짜뉴스', '주작', '사기', '거짓말', '허위', '선동'] if k in c]
    return len(detected), list(set(detected))

def run_forensic_main(url):
    st.session_state["debug_logs"] = []
    progress_text = "분석 시작 중..."
    my_bar = st.progress(0, text=progress_text)
    
    db_count, db_truth, db_fake = train_dynamic_vector_engine()
    
    my_bar.progress(10, text="1단계: 영상 자막 및 댓글 수집 중...")
    vid = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if vid: vid = vid.group(1)

    with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', ''); uploader = info.get('uploader', '')
            tags = info.get('tags', []); desc = info.get('description', '')
            
            trans, t_status = fetch_real_transcript(info)
            full_text = trans if trans else desc
            summary = summarize_transcript(full_text, title)
            top_transcript_keywords = extract_top_keywords_from_transcript(full_text)
            
            my_bar.progress(30, text="2단계: AI 수사관(Mistral)이 검색 키워드 추출 중...")
            query, source = get_mistral_search_keywords(title, full_text)

            my_bar.progress(50, text="3단계: 뉴스 크롤링 및 딥 웹 탐색 중...")
            is_official = check_is_official(uploader)
            is_ai, ai_msg = detect_ai_content(info)
            hashtag_display = ", ".join([f"#{t}" for t in tags]) if tags else "해시태그 없음"
            abuse_score, abuse_msg = check_tag_abuse(title, tags, uploader)
            agitation = count_sensational_words(full_text + title)
            
            ts, fs = vector_engine.analyze_position(query + " " + title)
            t_impact = int(ts * 30) * -1; f_impact = int(fs * 30)

            news_items = fetch_news_regex(query)
            news_ev = []; max_match = 0
            
            my_bar.progress(70, text="4단계: 뉴스 본문 정밀 대조 중...")
            for idx, item in enumerate(news_items[:3]):
                ai_s, ai_r, source_type, evidence_text, real_url = deep_verify_news(summary, item['link'], item['desc'])
                if ai_s > max_match: max_match = ai_s
                
                status_icon = "🟢" if ai_s >= 70 else "🔴" if ai_s < 30 else "🟡"
                news_ev.append({
                    "뉴스 제목": item['title'],
                    "일치도": f"{status_icon} {ai_s}%",
                    "최종 점수": f"{ai_s}%",
                    "분석 근거": ai_r,
                    "비고": f"[{source_type}] {len(evidence_text)}자 분석",
                    "원문": real_url
                })
            
            if not news_ev: news_score = 0
            else:
                if max_match >= 70: news_score = -30 
                elif max_match >= 50: news_score = -10
                else: news_score = 10 

            cmts, c_status = fetch_comments_via_api(vid)
            top_kw, rel_score, rel_msg = analyze_comment_relevance(cmts, title + " " + full_text)
            red_cnt, red_list = check_red_flags(cmts)
            
            silent_penalty = 0; is_silent = (len(news_ev) == 0)
            if is_silent:
                if any(k in title for k in CRITICAL_STATE_KEYWORDS): silent_penalty = 10
                elif agitation >= 3: silent_penalty = 20
            
            if is_official: news_score = -50; silent_penalty = 0
            sent_score = 0 
            
            clickbait = 10 if any(w in title for w in ['충격','경악','폭로']) else -5
            
            algo_base_score = 50 + t_impact + f_impact + news_score + sent_score + clickbait + abuse_score + silent_penalty
            
            my_bar.progress(90, text="5단계: AI 판사(Mistral) 최종 판결 중...")
            ai_judge_score, ai_judge_reason = get_mistral_verdict_final(title, full_text, news_ev)
            
            if t_impact == 0 and f_impact == 0 and is_silent:
                ai_judge_score = int((ai_judge_score + 50) / 2)
            
            final_prob = int((algo_base_score * WEIGHT_ALGO) + (ai_judge_score * WEIGHT_AI))
            final_prob = max(1, min(99, final_prob))
            
            save_analysis(uploader, title, final_prob, url, query)
            my_bar.empty()

            st.subheader("🕵️ Dual-Engine Analysis Result (Mistral Powered)")
            col_a, col_b, col_c = st.columns(3)
            with col_a: 
                st.metric("최종 가짜뉴스 확률", f"{final_prob}%", delta=f"AI Judge: {ai_judge_score}pt")
            with col_b:
                icon = "🟢" if final_prob < 30 else "🔴" if final_prob > 60 else "🟠"
                verdict = "안전 (Verified)" if final_prob < 30 else "위험 (Fake/Bias)" if final_prob > 60 else "주의 (Caution)"
                st.metric("종합 AI 판정", f"{icon} {verdict}")
            with col_c: 
                st.metric("AI Intelligence Level", f"{db_count} Nodes", delta="Hybrid Active")
            
            st.divider()
            st.subheader("🧠 Intelligence Map")
            render_intelligence_distribution(final_prob)

            if is_ai: st.warning(f"🤖 **AI 생성 콘텐츠 감지됨**: {ai_msg}")
            if is_official: st.success(f"🛡️ **공식 언론사 채널({uploader})입니다.**")

            st.divider()
            col1, col2 = st.columns([1, 1.4])
            with col1:
                st.write("**[영상 상세 정보]**")
                st.table(pd.DataFrame({"항목": ["영상 제목", "채널명", "조회수", "해시태그"], "내용": [title, uploader, f"{info.get('view_count',0):,}회", hashtag_display]}))
                st.info(f"🎯 **Investigator (Mistral A) 추출 검색어**: {query}")
                with st.container(border=True):
                    st.markdown("📝 **영상 내용 요약**")
                    st.write(summary)
                
                st.write("**[Score Breakdown]**")
                render_score_breakdown([
                    ["🏁 기본 중립 점수 (Base Score)", 50, "모든 분석은 50점(중립)에서 시작"],
                    ["진실 데이터 맥락", t_impact, "내부 DB 진실 데이터와 유사성"],
                    ["가짜 패턴 맥락", f_impact, "내부 DB 가짜 데이터와 유사성"],
                    ["뉴스 매칭 상태", news_score, "Deep-Crawler 정밀 대조 결과"],
                    ["여론/제목/태그 가감", sent_score + clickbait + abuse_score, ""],
                    ["-----------------", "", ""],
                    ["⚖️ AI Judge Score (40%)", ai_judge_score, "Mistral 종합 추론"]
                ])

            with col2:
                st.subheader("📊 5대 정밀 분석 증거")
                
                st.markdown("**[증거 0] Semantic Vector Space (Internal DB)**")
                colored_progress_bar("✅ 진실 영역 근접도", ts, "#2ecc71")
                colored_progress_bar("🚨 거짓 영역 근접도", fs, "#e74c3c")
                st.write("---")

                st.markdown(f"**[증거 1] 뉴스 교차 대조 (Deep-Web Crawler)**")
                if news_ev:
                    st.dataframe(
                        pd.DataFrame(news_ev),
                        column_config={
                            "원문": st.column_config.LinkColumn(label="링크", display_text="🔗 이동")
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    with st.expander("🔍 크롤링된 뉴스 본문 샘플 보기"):
                        for n in news_ev:
                            st.caption(f"**{n['뉴스 제목']}**: {n['비고']}")
                else: st.warning("🔍 관련 뉴스를 찾을 수 없습니다. (Silent Echo Risk)")
                    
                st.markdown("**[증거 2] 시청자 여론 심층 분석**")
                if cmts: st.table(pd.DataFrame([["최다 빈출 키워드", ", ".join(top_kw)], ["논란 감지 여부", f"{red_cnt}회"], ["주제 일치도", f"{rel_score}% ({rel_msg})"]], columns=["항목", "내용"]))
                
                st.markdown("**[증거 3] 자막 세만틱 심층 대조**")
                top_kw_str = ", ".join([f"{w}({c})" for w, c in top_transcript_keywords])
                st.table(pd.DataFrame([["영상 최다 언급 키워드", top_kw_str], ["제목 낚시어", "있음" if clickbait > 0 else "없음"], ["선동성 지수", f"{agitation}회"]], columns=["분석 항목", "판정 결과"]))
                
                st.markdown("**[증거 4] AI 최종 분석 판단 (Judge Verdict)**")
                with st.container(border=True):
                    st.write(f"⚖️ **판결:** {ai_judge_reason}")
                    st.caption(f"* Mistral 독립 추론 점수: {ai_judge_score}점 (Engine B)")

                reasons = []
                if final_prob >= 60:
                    reasons.append("🚨 **위험 감지**: AI 판사와 알고리즘 모두 이 영상의 주장을 의심하고 있습니다.")
                    if len(news_ev) == 0: reasons.append("🔇 **근거 부재**: 자극적인 주장에 비해 언론 보도가 전무합니다.")
                elif final_prob <= 30:
                    reasons.append("✅ **안전 판정**: 영상 내용이 주요 뉴스 보도와 일치하며, AI 추론 결과도 긍정적입니다.")
                else:
                    reasons.append("⚠️ **주의 요망**: 일부 과장된 표현이나 확인되지 않은 사실이 포함되어 있을 수 있습니다.")
                
                st.success(f"🔍 최종 분석 결과: **{final_prob}점**")
                for r in reasons: st.write(r)

        except Exception as e: st.error(f"오류: {e}")

# --- [UI Layout] ---
st.title("⚖️유튜브 가짜뉴스 판독기 (Mistral Edition)")

with st.container(border=True):
    st.markdown("### 🛡️ 법적 고지 및 책임 한계 (Disclaimer)\n본 서비스는 **인공지능(AI) 및 알고리즘 기반**으로 영상의 신뢰도를 분석하는 보조 도구입니다. \n분석 결과는 법적 효력이 없으며, 최종 판단의 책임은 사용자에게 있습니다.")
    st.markdown("* **Engine A (Investigator)**: Mistral Large/Small 기반 키워드 추출\n* **Engine B (Judge)**: 뉴스 본문 크롤링 및 정밀 대조")
    agree = st.checkbox("위 내용을 확인하였으며, 이에 동의합니다. (동의 시 분석 버튼 활성화)")

url_input = st.text_input("🔗 분석할 유튜브 URL")
if st.button("🚀 정밀 분석 시작", use_container_width=True, disabled=not agree):
    if url_input: run_forensic_main(url_input)
    else: st.warning("URL을 입력해주세요.")

st.divider()
st.subheader("🗂️ 학습 데이터 관리 (Cloud Knowledge Base)")
try:
    response = supabase.table("analysis_history").select("*").order("id", desc=True).execute()
    df = pd.DataFrame(response.data)
except: df = pd.DataFrame()

if not df.empty:
    if st.session_state["is_admin"]:
        df['Delete'] = False
        edited_df = st.data_editor(df[['Delete', 'id', 'analysis_date', 'video_title', 'fake_prob', 'keywords']], hide_index=True, use_container_width=True)
        if st.button("🗑️ 선택 항목 삭제", type="primary"):
            to_delete = edited_df[edited_df.Delete]
            if not to_delete.empty:
                for index, row in to_delete.iterrows(): supabase.table("analysis_history").delete().eq("id", row['id']).execute()
                st.success("삭제 완료!"); time.sleep(1); st.rerun()
    else:
        st.dataframe(df[['analysis_date', 'video_title', 'fake_prob', 'keywords']], hide_index=True, use_container_width=True)
else: st.info("데이터가 없습니다.")

st.write("")
# [관리자 전용 섹션]
with st.expander("🔐 관리자 접속 (Admin Access)"):
    if st.session_state["is_admin"]:
        st.success("관리자 권한 활성화됨")
        
        st.divider()
        st.subheader("🛠️ 시스템 상태 및 디버그 로그")
        
        st.write(f"**🤖 가용 모델 (Mistral):**")
        st.code(", ".join(AVAILABLE_MISTRAL_MODELS))
        
        if "debug_logs" in st.session_state and st.session_state["debug_logs"]:
            st.write(f"**📜 최근 실행 로그 ({len(st.session_state['debug_logs'])}건):**")
            log_text = "\n".join(st.session_state["debug_logs"])
            st.text_area("Logs", log_text, height=300)
        else:
            st.info("실행된 로그가 없습니다.")

        if st.button("로그아웃"):
            st.session_state["is_admin"] = False
            st.rerun()
    else:
        input_pwd = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if input_pwd == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True
                st.rerun()
            else:
                st.error("Access Denied")
