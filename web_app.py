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
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="Fact-Check Center v47.2 (Transcript+)", layout="wide", page_icon="⚖️")

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

# --- [관리자 인증] ---
if "is_admin" not in st.session_state: st.session_state["is_admin"] = False
with st.sidebar:
    st.header("🛡️ 관리자 메뉴")
    with st.form("login_form"):
        password_input = st.text_input("관리자 비밀번호", type="password")
        if st.form_submit_button("로그인"):
            if password_input == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True; st.rerun()
            else: st.session_state["is_admin"] = False; st.error("불일치")
    if st.session_state["is_admin"]:
        st.success("✅ 관리자 인증됨")
        if st.button("로그아웃"): st.session_state["is_admin"] = False; st.rerun()

# --- [상수] ---
WEIGHT_NEWS_DEFAULT = 45; WEIGHT_VECTOR = 35; WEIGHT_CONTENT = 15; WEIGHT_SENTIMENT_DEFAULT = 10
PENALTY_ABUSE = 20; PENALTY_MISMATCH = 30; PENALTY_NO_FACT = 25; PENALTY_SILENT_ECHO = 40

VITAL_KEYWORDS = ['위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '내란']
VIP_ENTITIES = ['윤석열', '대통령', '이재명', '한동훈', '김건희', '문재인', '박근혜', '이명박', '트럼프', '바이든', '푸틴', '젤렌스키', '시진핑', '정은', '이준석', '조국', '추미애', '홍준표', '유승민', '안철수', '손흥민', '이강인', '김민재', '류현진', '재용', '정의선', '최태원']
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']

STATIC_TRUTH_CORPUS = ["박나래 위장전입 무혐의", "임영웅 암표 대응", "정희원 저속노화", "대전 충남 통합", "선거 출마 선언"]
STATIC_FAKE_CORPUS = ["충격 폭로 경악", "긴급 속보 소름", "충격 발언 논란", "구속 영장 발부", "영상 유출", "계시 예언", "사형 집행", "위독설"]

class VectorEngine:
    def __init__(self): self.vocab = set(); self.truth_vectors = []; self.fake_vectors = []
    def tokenize(self, t): return re.findall(r'[가-힣]{2,}', t)
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

ve = VectorEngine()

def save_analysis(ch, ti, pr, url, kw):
    try: supabase.table("analysis_history").insert({"channel_name":ch, "video_title":ti, "fake_prob":pr, "analysis_date":datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url":url, "keywords":kw}).execute()
    except: pass

def train_ve():
    try:
        dt = [r['video_title'] for r in supabase.table("analysis_history").select("video_title").lt("fake_prob",30).execute().data]
        df = [r['video_title'] for r in supabase.table("analysis_history").select("video_title").gt("fake_prob",70).execute().data]
    except: dt, df = [], []
    ve.train(STATIC_TRUTH_CORPUS+dt, STATIC_FAKE_CORPUS+df)
    return len(STATIC_TRUTH_CORPUS+dt)+len(STATIC_FAKE_CORPUS+df)

# --- [UI Utils] ---
def colored_bar(label, val, color):
    st.markdown(f"<div style='margin-bottom:5px'><div style='display:flex;justify-content:space-between'><span>{label}</span><span style='color:{color};font-weight:bold'>{int(val*100)}%</span></div><div style='background:#eee;height:8px;border-radius:4px'><div style='background:{color};width:{val*100}%;height:100%;border-radius:4px'></div></div></div>", unsafe_allow_html=True)

def loading_seq(count):
    with st.status("🕵️ Semantic Core v47.2 가동...", expanded=True) as s:
        st.write(f"🧠 Intelligence Level: {count}")
        st.write("📝 자막 전체 데이터 수집 및 심층 분석 중...")
        time.sleep(0.5)
        st.write("✅ 분석 준비 완료!"); s.update(label="분석 완료!", state="complete", expanded=False)

# --- [Logic] ---
def extract_nouns(text):
    noise = ['충격','경악','속보','긴급','오늘','내일','지금','결국','뉴스','영상','대부분','이유','왜','있는','없는','하는','것','수','등','진짜','정말','알고보니','너무']
    return [n for n in re.findall(r'[가-힣A-Za-z0-9]{2,}', text) if n not in noise]

# 🌟 [v47.2 Upgrade] 자막에서 빈출 키워드 뽑기
def extract_top_keywords(text, top_n=3):
    nouns = extract_nouns(text)
    if not nouns: return []
    return [w for w, c in Counter(nouns).most_common(top_n)]

def generate_query(title, tags, transcript_keywords=[]):
    # 제목 + 태그 + 자막키워드 결합
    base_text = title + " " + " ".join([t.replace("#","") for t in tags])
    words = base_text.split()
    
    q_parts = []
    # 1. VIP/Vital 우선
    for w in words:
        if any(v in w for v in VITAL_KEYWORDS + VIP_ENTITIES): q_parts.append(w)
    
    # 2. 자막에서 뽑은 핵심 키워드 추가 (중복 방지)
    for kw in transcript_keywords:
        if kw not in q_parts and kw not in title:
            q_parts.append(kw)
    
    # 3. 없으면 제목 명사 사용
    if not q_parts: q_parts = extract_nouns(title)[:3]
    
    return " ".join(list(dict.fromkeys(q_parts))[:4]) # 최대 4단어

def summarize(text):
    if not text or len(text)<50: return "요약 정보 없음"
    sents = re.split(r'(?<=[.?!])\s+', text)
    freq = Counter(re.findall(r'[가-힣]{2,}', text))
    ranked = sorted([(i, s, sum(freq[w] for w in re.findall(r'[가-힣]{2,}',s))) for i,s in enumerate(sents) if 10<len(s)<150], key=lambda x:x[2], reverse=True)[:3]
    return " ".join([r[1] for r in sorted(ranked, key=lambda x:x[0])])

def check_official(uploader):
    return any(o in uploader.upper().replace(" ","") for o in OFFICIAL_CHANNELS)

def check_tags(title, tags, uploader):
    if check_official(uploader): return 0
    if not tags: return 0
    tn = set(extract_nouns(title)); tgn = set(t.replace("#","").split(":")[-1].strip() for t in tags)
    return 20 if len(tgn)>=2 and not tn.intersection(tgn) else 0

# 🌟 [v47.2 Upgrade] 자막 완전 수집 (중복 제거 로직 완화)
def fetch_transcript(info):
    try:
        url = None
        for k in ['subtitles','automatic_captions']:
            if k in info and 'ko' in info[k]:
                for f in info[k]['ko']:
                    if f['ext'] == 'vtt': url = f['url']; break
            if url: break
        
        if url:
            res = requests.get(url)
            if res.status_code == 200:
                clean = []
                for line in res.text.splitlines():
                    if '-->' not in line and 'WEBVTT' not in line and line.strip():
                        t = re.sub(r'<[^>]+>', '', line).strip()
                        # 중복 완화: 바로 앞 문장과 같을 때만 생략 (문맥 유지)
                        if t and (not clean or clean[-1] != t): 
                            clean.append(t)
                return " ".join(clean), "✅ 자막 전체 수집 완료"
    except: pass
    return None, "자막 없음"

def fetch_comments(vid):
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        res = requests.get(url, params={'part':'snippet', 'videoId':vid, 'key':YOUTUBE_API_KEY, 'maxResults':50, 'order':'relevance'})
        if res.status_code == 200:
            return [i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in res.json().get('items',[])], "성공"
    except: pass
    return [], "실패"

def calc_match(news_item, query_nouns, text):
    tn = set(extract_nouns(news_item['title'])); dn = set(extract_nouns(news_item['desc']))
    qn = set(query_nouns)
    t_score = 1.0 if len(qn & tn) >= 2 else 0.5 if len(qn & tn) >= 1 else 0
    c_cnt = sum(1 for n in dn if n in text)
    c_score = 1.0 if (len(dn)>0 and c_cnt/len(dn)>=0.3) else 0.5 if (len(dn)>0 and c_cnt/len(dn)>=0.15) else 0
    return int((t_score*0.3 + c_score*0.7)*100)

def analyze_comments(comments, text):
    if not comments: return [], 0, "분석 불가"
    cn = extract_nouns(" ".join(comments))
    top = Counter(cn).most_common(5)
    ctx = set(extract_nouns(text))
    match = sum(1 for w,c in top if w in ctx)
    score = int(match/len(top)*100) if top else 0
    msg = "✅ 일치" if score>=60 else "⚠️ 혼재" if score>=20 else "❌ 불일치"
    return [f"{w}({c})" for w,c in top], score, msg

def clean_html(raw): return BeautifulSoup(raw, "html.parser").get_text()

def run_main(url):
    intel = train_ve(); loading_seq(intel)
    vid = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if vid: vid = vid.group(1)
    
    with yt_dlp.YoutubeDL({'quiet':True, 'skip_download':True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get('title',''); uploader = info.get('uploader','')
            tags = info.get('tags',[]); desc = info.get('description','')
            
            # 1. 자막 수집 (업그레이드됨)
            trans, t_status = fetch_transcript(info)
            full_text = trans if trans else desc
            
            # 2. 자막에서 키워드 추출 (신규)
            trans_keywords = extract_top_keywords(full_text)
            
            # 3. 쿼리 생성 (자막 키워드 반영)
            query = generate_query(title, tags, trans_keywords)
            
            ts, fs = ve.analyze_position(query + " " + title)
            v_score = int(fs*10) - int(ts*10) # 가중치 조정
            
            # 뉴스 검색 (XML 방식 유지 - v47.1 기준)
            news_res = []; max_match = 0
            try:
                rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
                r = requests.get(rss, timeout=5)
                root = ET.fromstring(r.content)
                items = root.findall('.//item')
                for item in items[:3]:
                    nt = item.find('title').text
                    nd = clean_html(item.find('description').text) if item.find('description') is not None else ""
                    m = calc_match({'title':nt, 'desc':nd}, extract_nouns(query), full_text)
                    if m > max_match: max_match = m
                    news_res.append({"뉴스 제목": nt, "일치도": f"{m}%"})
            except: pass
            
            cmts, c_st = fetch_comments(vid)
            top_kw, rel_scr, rel_msg = analyze_comments(cmts, title + " " + full_text)
            red_cnt = sum(1 for c in cmts for k in ['가짜','주작','선동'] if k in c)
            
            n_score = 0; silent = 0; mismatch = 0
            is_silent = (len(news_res) == 0) or (len(news_res) > 0 and max_match < 20)
            agitation = full_text.count('충격') + full_text.count('경악')
            
            if is_silent:
                if agitation >= 3: silent = 40; v_score *= 2
                else: mismatch = 10
            elif red_cnt > 0:
                n_score = 25 if max_match < 60 else int((max_match/100)**2 * 65) * -1
            else:
                n_score = int((max_match/100)**2 * 45) * -1
                
            if check_official(uploader): n_score = -50; silent = 0; mismatch = 0
            
            tag_abuse = check_tags(title, tags, uploader)
            total = 50 + v_score + n_score + silent + mismatch + tag_abuse
            prob = max(5, min(99, total))
            
            save_analysis(uploader, title, prob, url, query)
            
            st.subheader("🕵️ 핵심 분석 지표")
            c1,c2,c3 = st.columns(3)
            c1.metric("가짜뉴스 확률", f"{prob}%", f"{total-50}")
            c2.metric("AI 판정", "🚨 위험" if prob>60 else "🟢 안전" if prob<30 else "🟠 주의")
            c3.metric("지능 레벨", intel)
            
            if silent: st.error("🔇 침묵의 메아리: 근거 부족")
            if check_official(uploader): st.success(f"🛡️ 공식 언론사({uploader})")
            
            st.divider()
            c1,c2 = st.columns([1,1])
            with c1:
                st.info(f"🎯 쿼리: {query}")
                st.caption(f"추출 키워드: {', '.join(trans_keywords)}")
                st.write("**영상 요약**"); st.caption(summarize(full_text))
                st.table(pd.DataFrame([["기본",50],["벡터",v_score],["뉴스",n_score],["페널티",silent+mismatch],["태그오용",tag_abuse]], columns=["항목","점수"]))
            with c2:
                colored_bar("진실", ts, "green"); colored_bar("거짓", fs, "red")
                st.write(f"**뉴스 ({len(news_res)}건)**"); st.table(news_res) if news_res else st.warning("뉴스 없음")
                st.write("**여론**"); st.caption(f"{rel_msg} (논란어 {red_cnt}회)")
                
        except Exception as e: st.error(f"분석 중 오류: {e}")

# --- [App] ---
st.title("⚖️ Triple-Evidence Intelligence Forensic v47.2")
url = st.text_input("🔗 유튜브 URL")
if st.button("🚀 분석 시작") and url: run_main(url)

st.divider()
st.subheader("🗂️ 학습 데이터 (Cloud)")
try:
    df = pd.DataFrame(supabase.table("analysis_history").select("*").order("id", desc=True).execute().data)
    if not df.empty:
        if st.session_state["is_admin"]:
            df['Delete'] = False
            cols = ['Delete'] + [c for c in df.columns if c != 'Delete']
            df = df[cols]
            ed = st.data_editor(df, column_config={"Delete": st.column_config.CheckboxColumn("삭제", default=False)}, disabled=["id","video_title","fake_prob"], hide_index=True, use_container_width=True)
            to_del = ed[ed.Delete]
            if not to_del.empty:
                if st.button(f"🗑️ {len(to_del)}건 삭제"):
                    for i, r in to_del.iterrows(): supabase.table("analysis_history").delete().eq("id", r['id']).execute()
                    st.success("삭제됨"); time.sleep(1); st.rerun()
        else: st.dataframe(df, hide_index=True)
    else: st.info("데이터 없음")
except: pass
