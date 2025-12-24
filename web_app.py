import streamlit as st
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
import altair as alt
import traceback

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="Fact-Check Center v54.0 (Masterpiece)", layout="wide", page_icon="⚖️")

# 🌟 Secrets 로드 (실패 시 에러 처리)
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except:
    st.error("❌ 필수 키(Secrets)가 설정되지 않았습니다. Streamlit 설정을 확인해주세요.")
    st.stop()

# DB 연결
from supabase import create_client
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    supabase = init_supabase()
except:
    st.error("❌ 데이터베이스 연결 실패")
    st.stop()

# --- [2. 핵심 분석 엔진 (Pure Logic NLP)] ---
# 무거운 AI 대신 정교한 규칙 기반 엔진 사용

VITAL_KEYWORDS = ['위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '내란', '간첩']
VIP_ENTITIES = ['윤석열', '대통령', '이재명', '한동훈', '김건희', '문재인', '박근혜', '이명박', '트럼프', '바이든', '푸틴', '젤렌스키', '시진핑', '정은', '이준석', '조국', '추미애', '홍준표', '유승민', '안철수', '손흥민', '이강인', '김민재', '류현진', '재용', '정의선', '최태원', '류중일', '감독', '조세호', '유재석', '장동민', '유호정', '이재룡', '임세령']
CRITICAL_STATE_KEYWORDS = ['별거', '이혼', '파경', '사망', '위독', '구속', '체포', '실형', '불화', '폭로', '충격', '논란', '중태', '심정지', '뇌사', '압수수색', '소환', '파산', '빚더미', '전과', '감옥', '간첩']
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아', 'JTBC News', 'SBS 뉴스', 'KBS News', 'MBCNEWS']

def normalize_korean_word(word):
    """한국어 조사 제거 (Regex)"""
    # 은/는/이/가/을/를/의/에/에서/로/으로/와/과/도/만...
    josa_pattern = r'(은|는|이|가|을|를|의|에|에서|로|으로|와|과|도|만|한테|에게|이랑|까지|부터|조차|마저|이라고|라는|다는)$'
    if len(word) >= 2:
        return re.sub(josa_pattern, '', word)
    return word

def extract_meaningful_tokens(text):
    """의미 있는 단어 추출"""
    # 한글만 추출
    raw_tokens = re.findall(r'[가-힣]{2,}', text)
    # 불용어(Stopwords)
    noise = ['충격', '경악', '속보', '긴급', '오늘', '내일', '지금', '결국', '뉴스', '영상', '대부분', '이유', '왜', '있는', '없는', '하는', '것', '수', '등', '진짜', '정말', '너무', '그냥', '이제', '사실', '국민', '우리', '대한민국', '여러분', '그리고', '그래서', '그러나', '솔직히', '무슨', '어떤']
    
    tokens = [normalize_korean_word(w) for w in raw_tokens]
    return [t for t in tokens if t not in noise and len(t) > 1]

def detect_subject_logic(title):
    """제목에서 주어(Subject) 추론"""
    tokens = extract_meaningful_tokens(title)
    
    # 1. VIP 리스트 매칭 (최우선)
    for vip in VIP_ENTITIES:
        if vip in title: return vip
    
    # 2. 호칭 기반 추론 ("OOO 회장", "OOO 씨")
    honorifics = ['회장', '의원', '대표', '대통령', '장관', '박사', '교수', '감독', '선수', '씨', '배우', '가수', '개그맨', '방송인']
    title_split = title.split()
    for i, word in enumerate(title_split):
        for hon in honorifics:
            if hon in word and i > 0:
                prev_word = normalize_korean_word(title_split[i-1])
                if len(prev_word) > 1: return prev_word
                
    # 3. 문장 맨 앞 명사 (확률 높음)
    if tokens: return tokens[0]
    return ""

def generate_smart_query(title, transcript):
    """뉴스 검색용 최적 쿼리 생성"""
    # 1. 주어 찾기
    subject = detect_subject_logic(title)
    
    # 2. 핵심 행위/사건 찾기 (제목과 자막의 교집합 중 가장 긴 단어)
    t_tokens = set(extract_meaningful_tokens(title))
    # 자막 앞부분만 사용하여 문맥 파악
    tr_tokens = set(extract_meaningful_tokens(transcript[:1000]))
    
    common = t_tokens.intersection(tr_tokens)
    # 주어 제외하고 나머지 중 가장 긴 단어 (구체적 사건일 확률 높음)
    actions = [w for w in common if w != subject]
    
    action = max(actions, key=len) if actions else ""
    
    # 3. Fallback: 교집합이 없으면 제목의 중요 단어 사용
    if not action:
        # 제목에서 치명적 키워드가 있으면 그걸 사용
        for crit in CRITICAL_STATE_KEYWORDS:
            if crit in title:
                action = crit
                break
    
    # 4. 최종 조합
    if subject and action:
        return f"{subject} {action}"
    elif subject:
        return f"{subject} {title.split()[-1]}" # 주어 + 제목 끝단어
    else:
        return " ".join(extract_meaningful_tokens(title)[:3])

# --- [3. 데이터 수집 및 분석 함수] ---
def fetch_real_transcript(info):
    try:
        url = None
        # 자동 자막 우선 탐색
        for key in ['subtitles', 'automatic_captions']:
            if key in info and 'ko' in info[key]:
                for fmt in info[key]['ko']:
                    if fmt['ext'] == 'vtt': url = fmt['url']; break
            if url: break
            
        if url:
            res = requests.get(url)
            if res.status_code == 200 and "#EXTM3U" not in res.text:
                clean = []
                for line in res.text.splitlines():
                    if '-->' not in line and 'WEBVTT' not in line and line.strip():
                        t = re.sub(r'<[^>]+>', '', line).strip()
                        if t and t not in clean: clean.append(t)
                return " ".join(clean)
    except: pass
    return info.get('description', '')

def fetch_news_regex(query):
    news_res = []
    try:
        # 구글 뉴스 RSS 사용 (가볍고 빠름)
        rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR&ceid=KR:ko"
        raw = requests.get(rss_url, timeout=5).text
        items = re.findall(r'<item>(.*?)</item>', raw, re.DOTALL)
        
        for item in items[:10]: # 상위 10개
            t = re.search(r'<title>(.*?)</title>', item)
            d = re.search(r'<description>(.*?)</description>', item) # RSS엔 설명이 없을 수 있음
            
            nt = t.group(1).replace("<![CDATA[", "").replace("]]>", "") if t else ""
            nd = clean_html_regex(d.group(1).replace("<![CDATA[", "").replace("]]>", "")) if d else ""
            
            # 출처 추출 (제목 뒤 ' - 언론사명')
            source = ""
            if " - " in nt:
                parts = nt.rsplit(" - ", 1)
                nt = parts[0]
                source = parts[1]
                
            news_res.append({'title': nt, 'desc': nd, 'source': source})
    except: pass
    return news_res

def clean_html_regex(text):
    return re.sub('<.*?>', '', text).strip()

def calculate_match_score(news_title, query, transcript, video_title):
    # 1. 쿼리 키워드 매칭
    q_tokens = set(extract_meaningful_tokens(query))
    n_tokens = set(extract_meaningful_tokens(news_title))
    
    match_cnt = len(q_tokens & n_tokens)
    base_score = 0
    
    if match_cnt >= 2: base_score = 80
    elif match_cnt == 1: base_score = 40
    
    # 2. Critical Check (치명적 키워드 불일치 시 0점)
    # 예: 영상엔 '사망'이 있는데 뉴스엔 없다? -> 0점
    for crit in CRITICAL_STATE_KEYWORDS:
        if crit in video_title and crit not in news_title:
            return 0
            
    return base_score

def summarize_text_simple(text):
    if not text: return "요약할 내용이 없습니다."
    sents = text.split('.')
    # 3문장만 추출
    return ". ".join([s.strip() for s in sents[:3] if s.strip()]) + "."

def save_analysis_history(channel, title, score, url, query):
    try:
        supabase.table("analysis_history").insert({
            "channel_name": channel,
            "video_title": title,
            "fake_prob": score,
            "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "video_url": url,
            "keywords": query
        }).execute()
    except: pass

def get_db_stats():
    try:
        res = supabase.table("analysis_history").select("fake_prob").execute()
        if res.data:
            df = pd.DataFrame(res.data)
            return len(df), len(df[df['fake_prob'] < 40]), len(df[df['fake_prob'] > 60]), df
    except: pass
    return 0, 0, 0, pd.DataFrame()

# --- [4. UI 컴포넌트] ---
def render_score_breakdown(items):
    # HTML Table로 점수 내역 이쁘게 표시
    rows = ""
    for label, score, note in items:
        color = "#ffcccc" if score > 0 else "#ccffcc" if score < 0 else "#f0f0f0"
        sign = "+" if score > 0 else ""
        rows += f"<tr><td style='padding:8px;'>{label}<br><span style='font-size:0.8em;color:gray'>{note}</span></td><td style='padding:8px;text-align:right;background-color:{color};font-weight:bold'>{sign}{score}</td></tr>"
    
    st.markdown(f"""
    <table style="width:100%; border-collapse:collapse; border:1px solid #ddd; font-size:14px;">
        <thead><tr style="background-color:#f9f9f9;"><th>분석 항목</th><th style="text-align:right">점수</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

def witty_loading(step):
    msgs = [
        "🧠 Pure Logic Engine 초기화 중...",
        "📡 영상 데이터 및 자막 추출 중...",
        "🔍 정교한 패턴 매칭 및 팩트 교차 검증 중...",
        "⚖️ 최종 판결문 작성 중..."
    ]
    with st.status("🕵️ 정밀 분석 진행 중...", expanded=True) as status:
        st.write(msgs[step])
        time.sleep(0.5)
        status.update(label="분석 완료!", state="complete", expanded=False)

# --- [5. 메인 앱 실행] ---
def main():
    # 사이드바
    with st.sidebar:
        st.header("🛡️ 관리자")
        if st.session_state.get("is_admin", False):
            st.success("로그인됨")
            if st.button("로그아웃"): st.session_state["is_admin"] = False; st.rerun()
        else:
            with st.form("login"):
                if st.form_submit_button("로그인"):
                    if st.text_input("PW", type="password") == ADMIN_PASSWORD:
                        st.session_state["is_admin"] = True; st.rerun()
                        
        st.divider()
        db_total, t_cnt, f_cnt, _ = get_db_stats()
        st.metric("누적 데이터", f"{db_total}건")
        st.caption(f"진실: {t_cnt} | 거짓: {f_cnt}")

    st.title("⚖️ Fact-Check Center v54.0")
    st.caption("🚀 Powered by **Pure Logic Engine** (Fast & Stable)")

    url_input = st.text_input("🔗 분석할 유튜브 URL 입력")
    
    if st.button("🚀 정밀 분석 시작", type="primary", use_container_width=True):
        if not url_input:
            st.warning("URL을 입력해주세요.")
            return

        witty_loading(0)
        
        # 1. 영상 정보 추출
        witty_loading(1)
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            try:
                info = ydl.extract_info(url_input, download=False)
                title = info.get('title', '')
                uploader = info.get('uploader', '')
                tags = info.get('tags', [])
                full_text = fetch_real_transcript(info)
            except Exception as e:
                st.error(f"영상 정보를 가져올 수 없습니다: {e}")
                return

        # 2. 분석 로직 수행
        witty_loading(2)
        
        # 쿼리 생성
        query = generate_smart_query(title, full_text)
        
        # 뉴스 검색
        news_items = fetch_news_regex(query)
        
        # 일치도 계산
        max_match_score = 0
        verified_news = []
        for item in news_items:
            s = calculate_match_score(item['title'], query, full_text, title)
            if s > max_match_score: max_match_score = s
            verified_news.append({'뉴스 제목': item['title'], '출처': item['source'], '일치도': f"{s}%"})
            
        # --- 점수 산정 (Scoring) ---
        base_score = 50
        details = []
        
        # A. 뉴스 검증 점수
        is_silent = (len(news_items) == 0) or (max_match_score < 30)
        has_critical = any(k in title for k in CRITICAL_STATE_KEYWORDS)
        
        news_score = 0
        news_note = ""
        
        if is_silent:
            if has_critical:
                news_score = 5 # 중립적 경고
                news_note = "⚠️ 미검증 위험 주장 (판단 보류)"
            else:
                news_score = 10
                news_note = "증거 불충분 (침묵)"
        else:
            if max_match_score >= 80:
                news_score = -45
                news_note = "✅ 뉴스 검증 완료 (팩트 일치)"
            elif max_match_score >= 40:
                news_score = -20
                news_note = "부분적 사실 확인"
            else:
                news_score = 10
                news_note = "낮은 연관성"
                
        details.append(("뉴스 교차 검증", news_score, news_note))
        
        # B. 공식 채널 보너스
        official_score = 0
        if any(o in uploader for o in OFFICIAL_CHANNELS):
            official_score = -50
            details.append(("공식 언론사", -50, "신뢰도 보장"))
            
        # C. 자극성 페널티
        agitation = sum(title.count(w) + full_text.count(w) for w in ['충격','경악','폭로','속보','긴급'])
        agitation_score = min(agitation * 5, 20)
        if agitation_score > 0:
            details.append(("자극적 표현", agitation_score, f"선동 키워드 {agitation}회"))
            
        # 최종
