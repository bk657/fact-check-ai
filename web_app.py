import streamlit as st
from supabase import create_client, Client
import re
import requests
import time
import random
import math
import json
import google.generativeai as genai
from datetime import datetime
from collections import Counter
import yt_dlp
import pandas as pd
import altair as alt

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="Fact-Check Center v69.0 (Single Shot)", layout="wide", page_icon="⚖️")

# 세션 상태 초기화
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

# 🌟 Secrets 로드
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("❌ 필수 키(API Key, DB Key, Password)가 설정되지 않았습니다.")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [2. 상수 정의] ---
WEIGHT_NEWS_DEFAULT = 45; WEIGHT_VECTOR = 35; WEIGHT_CONTENT = 15; WEIGHT_SENTIMENT_DEFAULT = 10
PENALTY_ABUSE = 20; PENALTY_MISMATCH = 30; PENALTY_NO_FACT = 25; PENALTY_SILENT_ECHO = 40

VITAL_KEYWORDS = ['위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '내란', '간첩']
CRITICAL_STATE_KEYWORDS = ['별거', '이혼', '파경', '사망', '위독', '구속', '체포', '실형', '불화', '폭로', '충격', '논란', '중태', '심정지', '뇌사', '압수수색', '소환', '파산', '빚더미', '전과', '감옥', '간첩']
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']

STATIC_TRUTH_CORPUS = ["박나래 위장전입 무혐의", "임영웅 암표 대응", "정희원 저속노화", "대전 충남 통합", "선거 출마 선언"]
STATIC_FAKE_CORPUS = ["충격 폭로 경악", "긴급 속보 소름", "충격 발언 논란", "구속 영장 발부", "영상 유출", "계시 예언", "사형 집행", "위독설"]

# --- [3. VectorEngine] ---
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

# --- [4. Gemini Logic (Single Shot)] ---
def get_gemini_response_stable(prompt_template, input_text):
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # 🚨 모델 순서: Flash -> Pro -> Legacy
    candidates = [
        ('gemini-1.5-flash', 30000), 
        ('gemini-1.5-pro', 20000), 
        ('gemini-pro', 5000)
    ]
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    error_logs = []
    
    for i, (model_name, char_limit) in enumerate(candidates):
        try:
            current_context = input_text[:char_limit]
            formatted_prompt = prompt_template.replace("{INPUT}", current_context)
            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(formatted_prompt, safety_settings=safety_settings)
            
            if response.text:
                return response.text.strip(), model_name, error_logs
                
        except Exception as e:
            err_msg = str(e)
            error_logs.append(f"{model_name}: {err_msg[:20]}...")
            time.sleep(2) # 2초 대기 후 다음 모델
            continue
            
    return None, "Fail", error_logs

def get_gemini_search_keywords(title, transcript):
    # 🚨 검색어 추출만 수행 (이것이 유일한 AI 호출)
    prompt_template = f"Extract ONE core search query (Nouns only). Input: {title} Transcript: {{INPUT}} Output: Query string only (Korean)."
    
    result, model_name, errors = get_gemini_response_stable(prompt_template, transcript)
    
    if result:
        return result, f"✨ Gemini ({model_name.replace('models/','')})"
    
    # 백업 로직
    tokens = re.findall(r'[가-힣]{2,}', title)
    cleaned = []
    for t in tokens:
        t = re.sub(r'(은|는|이|가|을|를|의)$', '', t)
        if len(t) > 1: cleaned.append(t)
    
    err_summary = " / ".join(errors) if errors else "Unknown"
    return " ".join(cleaned[:3]) if cleaned else title, f"🤖 Backup (Err: {err_summary[:30]}...)"

# --- [5. 유틸리티 함수] ---
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
        dt = [row['video_title'] for row in supabase.table("analysis_history").select("video_title").lt("fake_prob", 40).execute().data]
        df = [row['video_title'] for row in supabase.table("analysis_history").select("video_title").gt("fake_prob", 60).execute().data]
        vector_engine.train(STATIC_TRUTH_CORPUS + dt, STATIC_FAKE_CORPUS + df)
        return len(STATIC_TRUTH_CORPUS + dt) + len(STATIC_FAKE_CORPUS + df), len(dt), len(df)
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
    st.markdown(f"{style}<table class='score-table'><thead><tr><th>분석 항목 (Silent Echo Protocol)</th><th style='text-align: right;'>변동</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

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
    return (PENALTY_ABUSE, "🚨 심각 (불일치)") if not tn.intersection(tgn) else (0, "양호")

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
            
            nt = t.group(1).replace("<![CDATA[", "").replace("]]>", "") if t else ""
            nd = clean_html_regex(d.group(1).replace("<![CDATA[", "").replace("]]>", "")) if d else ""
            nl = l.group(1) if l else "#"
            
            news_res.append({'title': nt, 'desc': nd, 'link': nl})
    except: pass
    return news_res

def extract_top_keywords_from_transcript(text, top_n=5):
    if not text: return []
    tokens = extract_meaningful_tokens(text)
    return Counter(tokens).most_common(top_n)

def calculate_dual_match(news_item, query_nouns, video_summary):
    news_title_tokens = set(extract_meaningful_tokens(news_item.get('title', '')))
    qn = set(query_nouns)
    title_match_score = 0
    if len(qn & news_title_tokens) >= 2: title_match_score = 100
    elif len(qn & news_title_tokens) >= 1: title_match_score = 50
    
    news_desc = news_item.get('desc', '')
    content_sim_score = 0
    if news_desc and video_summary:
        sim = vector_engine.compute_content_similarity(video_summary, news_desc)
        content_sim_score = int(sim * 100)
    
    final_score = int((title_match_score * 0.4) + (content_sim_score * 0.6))
    for critical in CRITICAL_STATE_KEYWORDS:
        if critical in query_nouns and critical not in news_title_tokens:
            final_score = 0
            
    return title_match_score, content_sim_score, final_score

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
    messages = [f"🧠 [Intelligence: {total}] 집단 지성 로드 중...", f"📚 학습된 진실/거짓 데이터 로드 완료", "🚀 정밀 분석 엔진 가동"]
    with st.status("🕵️ Hybrid Fact-Check Engine v69.0...", expanded=True) as status:
        for msg in messages: st.write(msg); time.sleep(0.3)
        status.update(label="분석 준비 완료", state="complete", expanded=False)

# --- [Main Execution] ---
def run_forensic_main(url):
    total_intelligence, t_cnt, f_cnt = train_dynamic_vector_engine()
    witty_loading_sequence(total_intelligence, t_cnt, f_cnt)
    
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
            is_official = check_is_official(uploader)
            is_ai, ai_msg = detect_ai_content(info)
            w_news = 70 if is_ai else WEIGHT_NEWS_DEFAULT
            w_vec = 10 if is_ai else WEIGHT_VECTOR
            
            # Gemini Call 1 (Single Shot)
            query, source = get_gemini_search_keywords(title, full_text)
            
            hashtag_display = ", ".join([f"#{t}" for t in tags]) if tags else "해시태그 없음"
            abuse_score, abuse_msg = check_tag_abuse(title, tags, uploader)
            agitation = count_sensational_words(full_text + title)
            
            ts, fs = vector_engine.analyze_position(query + " " + title)
            t_impact = int(ts * w_vec) * -1; f_impact = int(fs * w_vec)

            news_items = fetch_news_regex(query)
            news_ev = []; max_match = 0
            mismatch_count = 0
            
            for item in news_items:
                t_score, c_score, final = calculate_dual_match(item, extract_meaningful_tokens(query), summary)
                if final > max_match: max_match = final
                if final < 20: mismatch_count += 1
                
                news_ev.append({
                    "뉴스 제목": item['title'],
                    "제목 일치": f"{t_score}%",
                    "내용 유사": f"{c_score}%",
                    "최종 점수": f"{final}%",
                    "기사 링크": item['link']
                })
            
            # 뉴스 점수 계산
            if not news_ev:
                news_score = 0
            else:
                if max_match >= 60: news_score = int((max_match / 100) * w_news) * -1
                else:
                    if mismatch_count >= len(news_ev) * 0.5: news_score = 20
                    else: news_score = 0

            cmts, c_status = fetch_comments_via_api(vid)
            top_kw, rel_score, rel_msg = analyze_comment_relevance(cmts, title + " " + full_text)
            red_cnt, red_list = check_red_flags(cmts)
            is_controversial = red_cnt > 0
            
            w_news = 65 if is_controversial else w_news
            
            silent_penalty = 0; mismatch_penalty = 0
            is_silent = (len(news_ev) == 0)
            has_critical_claim = any(k in title for k in CRITICAL_STATE_KEYWORDS)
            
            is_gray_zone = False
            
            if is_silent:
                if has_critical_claim:
                    silent_penalty = 5; t_impact = 0; f_impact = 0; is_gray_zone = True
                elif agitation >= 3:
                    silent_penalty = PENALTY_SILENT_ECHO
                    t_impact *= 2; f_impact *= 2
                else:
                    mismatch_penalty = 10
            
            if not is_silent and mismatch_count > 0 and max_match < 30:
                mismatch_penalty = PENALTY_MISMATCH
                
            if is_official: news_score = -50; mismatch_penalty = 0; silent_penalty = 0
            
            sent_score = 0
            if cmts and not is_controversial:
                neg = sum(1 for c in cmts for k in ['가짜','선동'] if k in c) / len(cmts)
                sent_score = int(neg * 10)
                
            clickbait = 10 if any(w in title for w in ['충격','경악','폭로']) else -5
            
            algo_score = 50 + t_impact + f_impact + news_score + sent_score + clickbait + abuse_score + mismatch_penalty + silent_penalty
