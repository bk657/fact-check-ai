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
st.set_page_config(page_title="Fact-Check Center v90.0 (Real Survivor)", layout="wide", page_icon="🧬")

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

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

# --- [2. 모델 자동 탐색기 (Auto-Discovery)] ---
@st.cache_data(ttl=3600) # 1시간마다 갱신
def get_all_available_models(api_key):
    genai.configure(api_key=api_key)
    models = []
    try:
        # 구글 API에 등록된 모든 모델을 긁어옵니다.
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # v1beta 모델 등 이름 정리
                model_name = m.name.replace("models/", "")
                models.append(model_name)
    except Exception as e:
        # API 호출 실패 시 비상용 하드코딩 리스트
        return ["gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-2.0-flash", "gemini-1.5-flash"]
    
    # [우선순위 정렬] Lite > Flash > Pro 순서로 정렬 (속도 및 쿼터 최적화)
    # 1. Lite가 들어간 것 우선
    # 2. 그 다음 Flash가 들어간 것
    # 3. 나머지는 뒤로
    def sort_key(name):
        if 'lite' in name: return 0
        if 'flash' in name: return 1
        return 2
    
    models.sort(key=sort_key)
    return models

# --- [3. 상수 정의] ---
WEIGHT_ALGO = 0.6
WEIGHT_AI = 0.4

VITAL_KEYWORDS = ['위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '내란', '간첩']
CRITICAL_STATE_KEYWORDS = ['별거', '이혼', '파경', '사망', '위독', '구속', '체포', '실형', '불화', '폭로', '충격', '논란', '중태', '심정지', '뇌사', '압수수색', '소환', '파산', '빚더미', '전과', '감옥', '간첩']
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']

STATIC_TRUTH_CORPUS = ["박나래 위장전입 무혐의", "임영웅 암표 대응", "정희원 저속노화", "대전 충남 통합", "선거 출마 선언"]
STATIC_FAKE_CORPUS = ["충격 폭로 경악", "긴급 속보 소름", "충격 발언 논란", "구속 영장 발부", "영상 유출", "계시 예언", "사형 집행", "위독설"]

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

# --- [5. Gemini Logic (The Real Survivor)] ---

# 🚨 안전 설정
safety_settings_none = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# ⚔️ [핵심] 무적의 생존자 호출 함수
def call_gemini_survivor(api_key, prompt, is_json=False):
    genai.configure(api_key=api_key)
    generation_config = {"response_mime_type": "application/json"} if is_json else {}
    
    # 1. 현재 사용 가능한 모든 모델 리스트를 가져옵니다.
    all_models = get_all_available_models(api_key)
    
    logs = []
    
    # 2. 리스트를 순회하며 살아있는 모델을 찾습니다.
    for model_name in all_models:
        try:
            model = genai.GenerativeModel(model_name, generation_config=generation_config)
            # 아주 짧은 타임아웃은 두지 않음 (생성 시간 필요)
            response = model.generate_content(prompt, safety_settings=safety_settings_none)
            
            if response.text:
                # 성공하면 로그에 남기고 리턴
                return response.text, model_name, logs
                
        except Exception as e:
            # 실패하면 로그에 남기고 다음 모델로 넘어감
            logs.append(f"❌ {model_name}: {str(e)[:50]}...")
            time.sleep(0.2) # 과부하 방지용 미세 딜레이
            continue
            
    # 3. 모든 모델이 다 죽었을 때 (최악의 경우)
    return None, "All Failed", logs

# [Engine A] 수사관: 키워드 추출
def get_gemini_search_keywords(title, transcript):
    # 자막이 너무 길면 모델이 힘들어하므로 앞부분만 (그러나 충분히 길게)
    context_data = transcript[:15000] 
    
    prompt = f"""
    You are a Fact-Check Investigator.
    
    [Input]
    Title: {title}
    Transcript (Partial): {context_data}
    
    [Task]
    Extract ONE specific search query for Google News.
    
    [Rules]
    1. **IGNORE** generic terms (Vlog, Mukbang, Daily life).
    2. **FOCUS** on specific Proper Nouns (Person's Name, Drug Name, Company Name, Crime Type).
    3. If the video mentions a specific scandal or death, include those keywords.
    4. **Output:** ONLY the Korean search query string (2-4 words).
    """
    
    result_text, model_used, logs = call_gemini_survivor(GOOGLE_API_KEY_A, prompt)
    
    # 사이드바에 로그 출력
    with st.sidebar.expander(f"🕵️ Key A (Investigator) Logs", expanded=False):
        for log in logs: st.write(log)
        if result_text: st.success(f"✅ Used: {model_used}")
        else: st.error("❌ All models failed")

    return (result_text.strip(), f"✨ {model_used}") if result_text else (title, "❌ Error")

# [크롤러] 뉴스 본문 수집
def scrape_news_content_robust(google_url):
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response = session.get(google_url, timeout=5, allow_redirects=True)
        final_url = response.url
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe']):
            tag.decompose()
            
        paragraphs = soup.find_all('p')
        clean_text = []
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 30: clean_text.append(text)
        
        full_text = " ".join(clean_text)
        if len(full_text) < 100: return None, final_url
        return full_text[:4000], final_url
    except:
        return None, google_url

# [Engine B] 판사: 뉴스 정밀 대조
def deep_verify_news(video_summary, news_url, news_snippet):
    scraped_text, real_url = scrape_news_content_robust(news_url)
    evidence_text = scraped_text if scraped_text else news_snippet
    source_type = "Full Article" if scraped_text else "Snippet Only"
    
    prompt = f"""
    Compare Video Summary vs News Evidence.
    
    [Video Summary]
    {video_summary[:2000]}
    
    [News Evidence ({source_type})]
    {evidence_text}
    
    [Task]
    Does the news confirm the video's claim?
    - Match: Score 90-100.
    - Related: Score 40-60.
    - Mismatch: Score 0-10.
    
    [Output JSON]
    {{ "score": <int>, "reason": "<short korean reason>" }}
    """
    
    result_text, model_used, logs = call_gemini_survivor(GOOGLE_API_KEY_B, prompt, is_json=True)
    
    try:
        res = json.loads(result_text)
        return res['score'], res['reason'], source_type, evidence_text, real_url
    except:
        return 0, "Analysis Failed", "Error", "", news_url

# [Engine B] 최종 판결
def get_gemini_verdict_final(title, transcript, verified_news_list):
    news_summary = ""
    for item in verified_news_list:
        news_summary += f"- News: {item['뉴스 제목']} (Score: {item['최종 점수']}, Reason: {item['분석 근거']})\n"
    
    full_context = transcript[:30000]
    prompt = f"""
    You are a professional Fact-Check AI Judge.
    
    [Video Info]
    Title: {title}
    Transcript Summary: {full_context[:2000]}...
    
    [Cross-Checked Evidence]
    {news_summary}
    
    [Instruction]
    1. Verify truthfulness based on evidence.
    2. Reliable match -> Truth (0-30).
    3. Mismatch/No evidence -> Fake (70-100).
    4. MUST OUTPUT REASON IN KOREAN.
    
    [Output Format - JSON Only]
    {{ "score": <int>, "reason": "<한글 판결문>" }}
    """
    
    result_text, model_used, logs = call_gemini_survivor(GOOGLE_API_KEY_B, prompt, is_json=True)
    
    # 사이드바에 로그 출력
    with st.sidebar.expander(f"⚖️ Key B (Judge) Logs", expanded=False):
        for log in logs: st.write(log)
        if result_text: st.success(f"✅ Used: {model_used}")
        
    if result_text:
        try:
            data = json.loads(result_text)
            return data['score'], f"{data['reason']} (By {model_used})"
        except: return 50, f"JSON Error"
    return 50, f"Judge Failed"

# --- [6. 유틸리티 함수] ---
def normalize_korean_word(word):
    word = re.sub(r'[^가-힣0-9]', '', word)
    for j in ['은','는','이','가','을','를','의','에','에게','로','으로']:
        if word.endswith(j): return word[:-len(j)]
    return word

def extract_meaningful_tokens(text):
    raw = re.findall(r'[가-힣]{2,}', text)
    noise = ['충격','속보','긴급','오늘','지금','결국','뉴스','영상']
    return [normalize_korean_word(w) for w in raw if w not in noise]

def train_dynamic_vector_engine():
    try:
        res_t = supabase.table("analysis_history").select("video_title").lt("fake_prob", 40).execute()
        res_f = supabase.table("analysis_history").select("video_title").gt("fake_prob", 60).execute()
        
        dt = [row['video_title'] for row in res_t.data] if res_t.data else []
        df = [row['video_title'] for row in res_f.data] if res_f.data else []
        
        vector_engine.train(STATIC_TRUTH_CORPUS + dt, STATIC_FAKE_CORPUS + df)
        return len(dt) + len(df), len(dt), len(df)
    except: 
        vector_engine.train(STATIC_TRUTH_CORPUS, STATIC_FAKE_CORPUS)
        return 0, 0, 0

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
    if not text or len(text) < 50: return "⚠️ 요약할 자막 내용이 충분하지 않습니다."
    clean_text = re.sub(r'http\S+|#EXTM3U|#EXT-X-VERSION:3', '', text)
    clean_text = re.sub(r'\[.*?\]|[>]+', '', clean_text)
    sentences = re.split(r'(?<=[.?!])\s+', clean_text)
    if len(sentences) <= 3: return clean_text.strip()
    title_nouns = set(extract_meaningful_tokens(title))
    scored_sentences = []
    for i, sent in enumerate(sentences):
        if len(sent) < 15: continue
        score = 0
        sent_tokens = extract_meaningful_tokens(sent)
        score += len(sent_tokens)
        for n in sent_tokens:
            if n in title_nouns: score += 10
        if i < len(sentences) * 0.2: score += 3
        elif i > len(sentences) * 0.8: score += 2
        scored_sentences.append((i, sent, score))
    top_sentences = sorted(scored_sentences, key=lambda x:x[2], reverse=True)[:max_sentences]
    top_sentences.sort(key=lambda x:x[0])
    return " ".join([s[1] for s in top_sentences])

def clean_html_regex(text):
    if not text: return ""
    return re.sub('<.*?>', '', text).strip()

def detect_ai_content(info):
    is_ai, reasons = False, []
    text = (info.get('title', '') + " " + info.get('description', '') + " " + " ".join(info.get('tags', []))).lower()
    for kw in ['ai', 'artificial intelligence', 'chatgpt', 'deepfake', 'synthetic', '인공지능', '딥페이크', '가상인간']:
        if kw in text: is_ai = True; reasons.append(f"키워드 감지: {kw}"); break
    return is_ai, ", ".join(reasons)

def check_is_official(channel_name):
    norm_name = channel_name.upper().replace(" ", "")
    return any(o in norm_name for o in OFFICIAL_CHANNELS)

def count_sensational_words(text):
    return sum(text.count(w) for w in ['충격', '경악', '실체', '폭로', '난리', '속보', '긴급', '소름', 'ㄷㄷ', '진짜', '결국', '계시', '예언', '위독', '사망', '중태'])

def check_tag_abuse(title, hashtags, channel_name):
    if check_is_official(channel_name): return 0, "공식 채널 면제"
    if not hashtags: return 0, "해시태그 없음"
    tn = set(extract_meaningful_tokens(title)); tgn = set(h.replace("#", "").split(":")[-1].strip() for h in hashtags)
    if len(tgn) < 2: return 0, "양호"
    return (20, "🚨 심각 (불일치)") if not tn.intersection(tgn) else (0, "양호")

def fetch_real_transcript(info_dict):
    try:
        url = None
        for key in ['subtitles', 'automatic_captions']:
            if key in info_dict and 'ko' in info_dict[key]:
                for fmt in info_dict[key]['ko']:
                    if fmt['ext'] == 'vtt': url = fmt['url']; break
            if url: break
        if url:
            res = requests.get(url)
            if res.status_code == 200:
                content = res.text
                if "#EXTM3U" in content: return None, "자막 포맷 오류"
                clean = []
                for line in content.splitlines():
                    if '-->' not in line and 'WEBVTT' not in line and line.strip():
                        t = re.sub(r'<[^>]+>', '', line).strip()
                        if t and t not in clean: clean.append(t)
                full_text = " ".join(clean)
                return full_text, f"✅ 전체 자막 수집 완료 (총 {len(full_text):,}자)"
    except: pass
    return None, "자막 다운로드 실패"

def fetch_comments_via_api(video_id):
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        res = requests.get(url, params={'part': 'snippet', 'videoId': video_id, 'key': YOUTUBE_API_KEY, 'maxResults': 50, 'order': 'relevance'})
        if res.status_code == 200:
            items = [i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in res.json().get('items', [])]
            return items, f"✅ API 수집 성공 (Top {len(items)})"
    except: pass
    return [], "❌ API 통신 실패"

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
            nt = t.group(1).replace("<![CDATA[", "").replace("]]>", "") if t else "제목 없음"
            nd = clean_html_regex(d.group(1).replace("<![CDATA[", "").replace("]]>", "")) if d else "내용 없음"
            nl = l.group(1).strip() if l else ""
            news_res.append({'title': nt, 'desc': nd, 'link': nl})
    except: pass
    return news_res

def extract_top_keywords_from_transcript(text, top_n=5):
    if not text: return []
    tokens = extract_meaningful_tokens(text)
    return Counter(tokens).most_common(top_n)

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

def witty_loading_sequence(total, t_cnt, f_cnt):
    messages = [f"🧠 [Intelligence: {total}] 집단 지성 로드 중...", f"🔑 Twin-Gemini Protocol 활성화...", "🚀 수사관(Investigator) 및 판사(Judge) 엔진 가동"]
    with st.status("🕵️ Dual-Engine Fact-Check v90.0...", expanded=True) as status:
        for msg in messages: st.write(msg); time.sleep(0.3)
        status.update(label="분석 준비 완료", state="complete", expanded=False)

def run_forensic_main(url):
    db_count, db_truth, db_fake = train_dynamic_vector_engine()
    witty_loading_sequence(db_count, 0, 0)
    
    vid = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if vid: vid = vid.group(1)

    with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', ''); uploader = info.get('uploader', '')
            tags = info.get('tags', []); desc = info.get('description', '')
            
            # 사이드바 초기화
            st.sidebar.title("🤖 AI Model Status")
            avail_models = get_all_available_models(GOOGLE_API_KEY_A)
            st.sidebar.success(f"Detected {len(avail_models)} Active Models")
            with st.sidebar.expander("Show Available Models"):
                st.write(avail_models)
                
            trans, t_status = fetch_real_transcript(info)
            full_text = trans if trans else desc
            summary = summarize_transcript(full_text, title)
            top_transcript_keywords = extract_top_keywords_from_transcript(full_text)
            
            query, source = get_gemini_search_keywords(title, full_text)

            is_official = check_is_official(uploader)
            is_ai, ai_msg = detect_ai_content(info)
            hashtag_display = ", ".join([f"#{t}" for t in tags]) if tags else "해시태그 없음"
            abuse_score, abuse_msg = check_tag_abuse(title, tags, uploader)
            agitation = count_sensational_words(full_text + title)
            
            ts, fs = vector_engine.analyze_position(query + " " + title)
            t_impact = int(ts * 30) * -1; f_impact = int(fs * 30)

            news_items = fetch_news_regex(query)
            news_ev = []; max_match = 0
            
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
            if cmts and red_cnt == 0:
                neg = sum(1 for c in cmts for k in ['가짜','선동'] if k in c) / len(cmts)
                sent_score = int(neg * 10)
            
            clickbait = 10 if any(w in title for w in ['충격','경악','폭로']) else -5
            
            algo_base_score = 50 + t_impact + f_impact + news_score + sent_score + clickbait + abuse_score + silent_penalty
            
            ai_judge_score, ai_judge_reason = get_gemini_verdict_final(title, full_text, news_ev)
            
            if t_impact == 0 and f_impact == 0 and is_silent:
                ai_judge_score = int((ai_judge_score + 50) / 2)
            
            final_prob = int((algo_base_score * WEIGHT_ALGO) + (ai_judge_score * WEIGHT_AI))
            final_prob = max(1, min(99, final_prob))
            
            save_analysis(uploader, title, final_prob, url, query)

            st.subheader("🕵️ Dual-Engine Analysis Result")
            col_a, col_b, col_c = st.columns(3)
            with col_a: 
                st.metric("최종 가짜뉴스 확률", f"{final_prob}%", delta=f"AI Judge: {ai_judge_score}pt")
            with col_b:
                icon = "🟢" if final_prob < 30 else "🔴" if final_prob > 60 else "🟠"
                verdict = "안전 (Verified)" if final_prob < 30 else "위험 (Fake/Bias)" if final_prob > 60 else "주의 (Caution)"
                st.metric("종합 AI 판정", f"{icon} {verdict}")
            with col_c: 
                st.metric("AI Intelligence Level", f"{db_count} Nodes", delta="Twin-Engine Active")
            
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
                st.info(f"🎯 **Investigator (Key A) 추출 검색어**: {query}")
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
                    ["⚖️ AI Judge Score (40%)", ai_judge_score, "Gemini 종합 추론"]
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
                    st.caption(f"* Gemini 독립 추론 점수: {ai_judge_score}점 (Key B)")

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
st.title("⚖️ Fact-Check Center v90.0 (Real Survivor)")

with st.container(border=True):
    st.markdown("### 🛡️ 법적 고지 및 책임 한계 (Disclaimer)\n본 서비스는 **인공지능(AI) 및 알고리즘 기반**으로 영상의 신뢰도를 분석하는 보조 도구입니다. \n분석 결과는 법적 효력이 없으며, 최종 판단의 책임은 사용자에게 있습니다.")
    st.markdown("* **Engine A (Investigator)**: 정밀 키워드 추출 (Real Survivor Mode)\n* **Engine B (Judge)**: 뉴스 본문 크롤링 및 정밀 대조 (Deep-Web Crawler)")
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
with st.expander("🔐 관리자 접속 (Admin Access)"):
    if st.session_state["is_admin"]:
        st.success("관리자 권한 활성화됨")
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
