import streamlit as st
from supabase import create_client, Client
import re
import requests
import time
import random
import math
from datetime import datetime
from collections import Counter
import yt_dlp
import pandas as pd
from bs4 import BeautifulSoup 

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="Fact-Check Center v47.1 (Final Stable)", layout="wide", page_icon="⚖️")

# 🌟 Secrets
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except:
    st.error("❌ 필수 키(API Key, DB Key, Password)가 설정되지 않았습니다.")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [관리자 인증 로직] ---
if "is_admin" not in st.session_state: st.session_state["is_admin"] = False

with st.sidebar:
    st.header("🛡️ 관리자 메뉴")
    # v47.1의 Form 방식 유지
    with st.form("login_form"):
        password_input = st.text_input("관리자 비밀번호", type="password")
        submit_button = st.form_submit_button("로그인")
        if submit_button:
            if password_input == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True; st.rerun()
            else:
                st.session_state["is_admin"] = False; st.error("비밀번호 불일치")

    if st.session_state["is_admin"]:
        st.success("✅ 관리자 인증됨")
        if st.button("로그아웃"):
            st.session_state["is_admin"] = False; st.rerun()

# --- [상수 설정 (v47.1 기준)] ---
WEIGHT_NEWS_DEFAULT = 45       
WEIGHT_VECTOR = 35     
WEIGHT_CONTENT = 15    
WEIGHT_SENTIMENT_DEFAULT = 10  
PENALTY_ABUSE = 20     
PENALTY_MISMATCH = 30
PENALTY_NO_FACT = 25
PENALTY_SILENT_ECHO = 40  

VITAL_KEYWORDS = ['위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '쓰러져', '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '못넘긴다']
VIP_ENTITIES = ['윤석열', '대통령', '이재명', '한동훈', '김건희', '문재인', '박근혜', '이명박', '트럼프', '바이든', '푸틴', '젤렌스키', '시진핑', '정은', '이준석', '조국', '추미애', '홍준표', '유승민', '안철수', '손흥민', '이강인', '김민재', '류현진', '재용', '정의선', '최태원']
# v47.2에서 수정 요청했던 엄격한 공식 채널 리스트 적용
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선일보', '중앙일보', '동아일보', '한국일보', '국민일보', '서울신문', '세계일보', '문화일보', '매일경제', '한국경제', '서울경제', 'CHOSUN', 'JOONGANG', 'DONGA', 'HANKYOREH', 'KYUNGHYANG']

STATIC_TRUTH_CORPUS = ["박나래 위장전입 의혹 무혐의", "임영웅 콘서트 암표 대응", "정희원 교수 저속노화", "대전 충남 행정 통합", "선거 출마 선언", "강훈식 의원 출마설"]
STATIC_FAKE_CORPUS = ["충격 폭로 경악", "긴급 속보 소름", "충격 발언 논란", "구속 영장 발부", "영상 유출", "꿈속 계시 예언", "사형 선고 집행", "건강 악화 위독설"]

class VectorEngine:
    def __init__(self): self.vocab = set(); self.truth_vectors = []; self.fake_vectors = []
    def tokenize(self, text): return re.findall(r'[가-힣]{2,}', text)
    def train(self, t_corpus, f_corpus):
        for t in t_corpus + f_corpus: self.vocab.update(self.tokenize(t))
        self.vocab = sorted(list(self.vocab))
        self.truth_vectors = [self.text_to_vector(t) for t in t_corpus]
        self.fake_vectors = [self.text_to_vector(t) for t in f_corpus]
    def text_to_vector(self, text):
        c = Counter(self.tokenize(text)); return [c[w] for w in self.vocab]
    def cosine_similarity(self, v1, v2):
        dot = sum(a*b for a,b in zip(v1,v2)); mag = math.sqrt(sum(a*a for a in v1)) * math.sqrt(sum(b*b for b in v2))
        return dot/mag if mag>0 else 0
    def analyze_position(self, query):
        qv = self.text_to_vector(query)
        mt = max([self.cosine_similarity(qv, v) for v in self.truth_vectors] or [0])
        mf = max([self.cosine_similarity(qv, v) for v in self.fake_vectors] or [0])
        return mt, mf

vector_engine = VectorEngine()

def save_analysis(channel, title, prob, url, keywords):
    try: supabase.table("analysis_history").insert({"channel_name": channel, "video_title": title, "fake_prob": prob, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url": url, "keywords": keywords}).execute()
    except: pass

def train_dynamic_vector_engine():
    try:
        dt = [row['video_title'] for row in supabase.table("analysis_history").select("video_title").lt("fake_prob", 30).execute().data]
        df = [row['video_title'] for row in supabase.table("analysis_history").select("video_title").gt("fake_prob", 70).execute().data]
    except: dt, df = [], []
    vector_engine.train(STATIC_TRUTH_CORPUS + dt, STATIC_FAKE_CORPUS + df)
    return len(STATIC_TRUTH_CORPUS + dt) + len(STATIC_FAKE_CORPUS + df)

# --- [Helper Functions] ---
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

def witty_loading_sequence(count):
    messages = [f"🧠 [Intelligence Level: {count}] 누적 지식 로드 중...", "🔄 '주어(Modifier)' + '핵심어(Head)' 역방향 결합(Back-Merge) 중...", "🎯 문맥을 통합하여 완벽한 검색어(Contextual Query) 생성...", "🚀 위성이 유튜브 본사 상공을 지나가는 중..."]
    with st.status("🕵️ Context Merger v47.1 가동 중...", expanded=True) as status:
        for msg in messages: st.write(msg); time.sleep(0.4)
        st.write("✅ 분석 준비 완료!"); status.update(label="분석 완료!", state="complete", expanded=False)

def extract_nouns(text):
    noise = ['충격', '경악', '실체', '난리', '공개', '반응', '명단', '동영상', '사진', '집안', '속보', '단독', '결국', 'MBC', '뉴스', '이미지', '너무', '다른', '알고보니', 'ㄷㄷ', '진짜', '정말', '영상', '사람', '생각', '오늘밤', '오늘', '내일', '지금', '못넘긴다', '넘긴다', '이유', '왜', '안']
    return list(dict.fromkeys([n for n in re.findall(r'[가-힣]{2,}', text) if n not in noise]))

def generate_pinpoint_query(title, hashtags):
    clean_text = title + " " + " ".join([h.replace("#", "") for h in hashtags])
    words = clean_text.split()
    subject_chunk, object_word, vital_word = "", "", ""
    for vital in VITAL_KEYWORDS:
        if vital in clean_text: vital_word = vital; break
    for i, word in enumerate(words):
        match = re.match(r'([가-힣A-Za-z0-9]+)(은|는|이|가|을|를|에|에게|로서|로)', word)
        if match:
            noun, josa = match.group(1), match.group(2)
            if noun in ['오늘밤', '지금', '이유', '결국']: continue
            if not subject_chunk and josa in ['은', '는', '이', '가']:
                prev_noun = ""
                if i > 0:
                    prev_word = words[i-1]
                    if re.fullmatch(r'[가-힣A-Za-z0-9]+', prev_word) and prev_word not in VITAL_KEYWORDS + ['충격', '속보']: prev_noun = prev_word
                subject_chunk = f"{prev_noun} {noun}" if prev_noun else noun
            elif not object_word and josa in ['을', '를', '에', '에게', '로']:
                if noun not in VITAL_KEYWORDS and noun not in subject_chunk: object_word = noun
    query_parts = [p for p in [subject_chunk, object_word, vital_word] if p]
    if not subject_chunk: return " ".join(extract_nouns(title)[:3])
    return " ".join(query_parts)

def summarize_transcript(text):
    if not text or len(text) < 50: return "⚠️ 요약할 자막 내용이 충분하지 않습니다."
    sents = re.split(r'(?<=[.?!])\s+', text)
    if len(sents) <= 3: return text
    freq = Counter(re.findall(r'[가-힣]{2,}', text))
    ranked = sorted([(i, s, sum(freq[w] for w in re.findall(r'[가-힣]{2,}',s))/len(re.findall(r'[가-힣]{2,}',s) or [1])) for i,s in enumerate(sents) if 10<len(s)<150], key=lambda x:x[2], reverse=True)[:3]
    return f"📌 **핵심 요약**: {' '.join([r[1] for r in sorted(ranked, key=lambda x:x[0])])}"

def clean_html(raw_html): return BeautifulSoup(raw_html, "html.parser").get_text()

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
    tn = set(extract_nouns(title)); tgn = set(h.replace("#", "").split(":")[-1].strip() for h in hashtags)
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
                clean = []
                for line in res.text.splitlines():
                    if '-->' not in line and 'WEBVTT' not in line and line.strip():
                        t = re.sub(r'<[^>]+>', '', line).strip()
                        if t and t not in clean: clean.append(t)
                return " ".join(clean), "✅ 실제 자막 수집 성공"
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

def calculate_dual_match(news_item, query_nouns, transcript):
    tn = set(extract_nouns(news_item.get('title', ''))); dn = set(extract_nouns(news_item.get('desc', '')))
    qn = set(query_nouns)
    t_score = 1.0 if len(qn & tn) >= 2 else 0.5 if len(qn & tn) >= 1 else 0
    c_cnt = sum(1 for n in dn if n in transcript)
    c_score = 1.0 if (len(dn) > 0 and c_cnt/len(dn) >= 0.3) else 0.5 if (len(dn) > 0 and c_cnt/len(dn) >= 0.15) else 0
    return int((t_score * 0.3 + c_score * 0.7) * 100)

def analyze_comment_relevance(comments, context_text):
    if not comments: return [], 0, "분석 불가"
    cn = extract_nouns(" ".join(comments))
    top = Counter(cn).most_common(5)
    ctx = set(extract_nouns(context_text))
    match = sum(1 for w,c in top if w in ctx)
    score = int(match/len(top)*100) if top else 0
    msg = "✅ 주제 집중" if score >= 60 else "⚠️ 일부 관련" if score >= 20 else "❌ 무관"
    return [f"{w}({c})" for w, c in top], score, msg

def check_red_flags(comments):
    detected = [k for c in comments for k in ['가짜뉴스', '주작', '사기', '거짓말', '허위', '선동'] if k in c]
    return len(detected), list(set(detected))

# 🌟 [Fix] XML -> Regex 교체 (기능 변경 없음, 오직 에러 방지용)
def fetch_news_safe(query):
    news_res = []
    try:
        rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
        raw = requests.get(rss, timeout=5).text
        items = re.findall(r'<item>(.*?)</item>', raw, re.DOTALL)
        for item in items[:3]:
            t = re.search(r'<title>(.*?)</title>', item); d = re.search(r'<description>(.*?)</description>', item)
            nt = t.group(1).replace("<![CDATA[", "").replace("]]>", "") if t else ""
            nd = clean_html(d.group(1).replace("<![CDATA[", "").replace("]]>", "")) if d else ""
            news_res.append({'title': nt, 'desc': nd})
    except: pass
    return news_res

# 🌟 [Fix] 삭제 콜백 (삭제 기능 안정
