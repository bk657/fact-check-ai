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
st.set_page_config(page_title="Fact-Check Center v48.3", layout="wide", page_icon="⚖️")

# 🌟 Secrets
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except:
    st.error("❌ 필수 키 설정 오류 (Secrets)")
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
VITAL_KEYWORDS = ['위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '내란']
VIP_ENTITIES = ['윤석열', '대통령', '이재명', '한동훈', '김건희', '문재인', '박근혜', '이명박', '트럼프', '바이든', '푸틴', '젤렌스키', '시진핑', '정은', '이준석', '조국', '추미애', '홍준표', '유승민', '안철수', '손흥민', '이강인', '김민재', '류현진', '재용', '정의선', '최태원']
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선일보', '중앙일보', '동아일보', '한국일보', '국민일보', '서울신문', '세계일보', '문화일보', '매일경제', '한국경제', '서울경제', 'CHOSUN', 'JOONGANG', 'DONGA', 'HANKYOREH', 'KYUNGHYANG']
STATIC_TRUTH = ["박나래 위장전입 무혐의", "임영웅 암표 대응", "정희원 저속노화", "대전 충남 통합", "선거 출마 선언"]
STATIC_FAKE = ["충격 폭로 경악", "긴급 속보 소름", "충격 발언 논란", "구속 영장 발부", "영상 유출", "계시 예언", "사형 집행", "위독설"]

class VectorEngine:
    def __init__(self): self.vocab = set(); self.truth = []; self.fake = []
    def tokenize(self, t): return re.findall(r'[가-힣]{2,}', t)
    def train(self, t, f):
        for x in t+f: self.vocab.update(self.tokenize(x))
        self.vocab = sorted(list(self.vocab))
        self.truth = [self.vec(x) for x in t]; self.fake = [self.vec(x) for x in f]
    def vec(self, t):
        c = Counter(self.tokenize(t)); return [c[w] for w in self.vocab]
    def sim(self, v1, v2):
        dot = sum(a*b for a,b in zip(v1,v2)); mag = math.sqrt(sum(a*a for a in v1)) * math.sqrt(sum(b*b for b in v2))
        return dot/mag if mag>0 else 0
    def analyze(self, q):
        qv = self.vec(q); mt = max([self.sim(qv,x) for x in self.truth] or [0]); mf = max([self.sim(qv,x) for x in self.fake] or [0])
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
    ve.train(STATIC_TRUTH+dt, STATIC_FAKE+df)
    return len(STATIC_TRUTH+dt)+len(STATIC_FAKE+df)

# --- [UI Utils] ---
def colored_bar(label, val, color):
    st.markdown(f"<div style='margin-bottom:5px'><div style='display:flex;justify-content:space-between'><span>{label}</span><span style='color:{color};font-weight:bold'>{int(val*100)}%</span></div><div style='background:#eee;height:8px;border-radius:4px'><div style='background:{color};width:{val*100}%;height:100%;border-radius:4px'></div></div></div>", unsafe_allow_html=True)

def loading_seq(level):
    with st.status("🕵️ Forensic Core v48.3 가동...", expanded=True) as s:
        st.write(f"🧠 Intelligence Level: {level}"); time.sleep(0.3)
        st.write("🛡️ 1차 분석: 파싱 오류 방어 및 구문 분석..."); time.sleep(0.3)
        st.write("✅ 분석 준비 완료!"); s.update(label="분석 완료!", state="complete", expanded=False)

# --- [Logic] ---
def get_safe_text(element):
    if element is not None and element.text: return element.text.strip()
    return ""

def clean_html(raw):
    if not raw: return ""
    try: return BeautifulSoup(raw, "html.parser").get_text()
    except: return raw

def extract_nouns(text):
    noise = ['충격','경악','속보','긴급','오늘','내일','지금','결국','뉴스','영상','대부분','이유','왜','있는','없는','하는','것','수','등']
    return [n for n in re.findall(r'[가-힣A-Za-z0-9]{2,}', text) if n not in noise]

def generate_hybrid_query(title, tags, transcript):
    text = title + " " + " ".join([t.replace("#","") for t in tags])
    tn = extract_nouns(text); trn = extract_nouns(transcript if transcript else "")
    top_trn = [w for w,c in Counter(trn).most_common(3)]
    
    vip = [v for v in VIP_ENTITIES if v in text]
    vital = [v for v in VITAL_KEYWORDS if v in text]
    
    q = []
    if vip:
        q.extend(vip); q.extend(vital)
        for n in tn: 
            if n not in q and n not in VIP_ENTITIES: q.append(n); break
    else:
        q.extend(tn[:2])
        for n in top_trn:
            if n not in q: q.append(n); 
            if len(q)>=3: break
    return " ".join(q)

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
    tn = set(extract_nouns(title)); tgn = set()
    for t in tags: tgn.add(t.replace("#","").split(":")[-1].strip())
    # 점수(int)만 반환하도록 수정됨
    return 20 if len(tgn)>=2 and not tn.intersection(tgn) else 0

def fetch_transcript(info):
    try:
        url = None
        if 'subtitles' in info and 'ko' in info['subtitles']:
            for fmt in info['subtitles']['ko']: 
                if fmt['ext'] == 'vtt': url = fmt['url']; break
        if not url and 'automatic_captions' in info and 'ko' in info['automatic_captions']:
            for fmt in info['automatic_captions']['ko']: 
                if fmt['ext'] == 'vtt': url = fmt['url']; break
        
        if url:
            res = requests.get(url)
            if res.status_code == 200:
                clean = []
                for line in res.text.splitlines():
                    if '-->' not in line and line.strip() and not line.startswith('WEBVTT'):
                        t = re.sub(r'<[^>]+>', '', line).strip()
                        if t and t not in clean: clean.append(t)
                return " ".join(clean), "성공"
    except: pass
    return None, "실패"

def fetch_comments(vid):
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        res = requests.get(url, params={'part':'snippet', 'videoId':vid, 'key':YOUTUBE_API_KEY, 'maxResults':50, 'order':'relevance'})
        if res.status_code == 200:
            return [i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in res.json().get('items',[])], "성공"
    except: pass
    return [], "실패"

def calc_match(news_item, query_nouns, text):
    title_n = set(extract_nouns(news_item['title']))
    desc_n = set(extract_nouns(news_item['desc']))
    query_n = set(query_nouns)
    
    t_score = 1.0 if len(query_n & title_n) >= 2 else 0.5 if len(query_n & title_n) >= 1 else 0
    
    c_cnt = 0
    if desc_n:
        for n in desc_n: 
            if n in text: c_cnt += 1
        c_score = 1.0 if c_cnt/len(desc_n) > 0.3 else 0.5 if c_cnt/len(desc_n) > 0.15 else 0
    else: c_score = 0
    
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

def run_main(url):
    intel = train_ve(); loading_seq(intel)
    vid = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if vid: vid = vid.group(1)
    
    with yt_dlp.YoutubeDL({'quiet':True, 'skip_download':True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get('title',''); uploader = info.get('uploader','')
            tags = info.get('tags',[]); desc = info.get('description','')
            
            trans, t_status = fetch_transcript(info)
            full_text = trans if trans else desc
            query = generate_hybrid_query(title, tags, full_text)
            
            # 1. Vector
            ts, fs = ve.analyze(query + " " + title)
            v_score = int(fs*35) - int(ts*35)
            
            # 2. News
            news_res = []; max_match = 0; news_cnt = 0
            try:
                rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
                root = ET.fromstring(requests.get(rss, timeout=5).content)
                items = root.findall('.//item'); news_cnt = len(items)
                
                for item in items[:3]:
                    nt = get_safe_text(item.find('title'))
                    nd = clean_html(get_safe_text(item.find('description')))
                    m = calc_match({'title':nt, 'desc':nd}, extract_nouns(query), full_text)
                    if m > max_match: max_match = m
                    news_res.append({"뉴스 제목": nt, "일치도": f"{m}%"})
            except Exception: pass 
            
            # 3. Comments
            cmts, c_st = fetch_comments(vid)
            top_kw, rel_scr, rel_msg = analyze_comments(cmts, title + " " + full_text)
            red_cnt = sum(1 for c in cmts for k in ['가짜','주작','선동'] if k in c)
            
            # Scoring
            n_score = 0; silent = 0; mismatch = 0
            is_silent = (news_cnt == 0) or (news_cnt > 0 and max_match < 20)
            agitation = sum(full_text.count(w) for w in ['충격','경악','속보'])
            
            if is_silent:
                if agitation >= 3: silent = 40; v_score *= 2 # 침묵의 메아리
                else: mismatch = 10
            elif red_cnt > 0: # 논란
                if max_match < 60: n_score = 25
                else: n_score = int((max_match/100)**2 * 65) * -1
            else:
                n_score = int((max_match/100)**2 * 45) * -1
                
            if check_official(uploader): n_score = -50; silent = 0; mismatch = 0
            
            # 🌟 [수정] check_tags()[0] 제거 -> check_tags()
            tag_abuse_score = check_tags(title, tags, uploader)
            total = 50 + v_score + n_score + silent + mismatch + tag_abuse_score
            prob = max(5, min(99, total))
            
            save_analysis(uploader, title, prob, url, query)
            
            # Output
            st.subheader("🕵️ 핵심 분석 지표")
            c1,c2,c3 = st.columns(3)
            c1.metric("가짜뉴스 확률", f"{prob}%", f"{total-50}")
            c2.metric("AI 판정", "🚨 위험" if prob>60 else "🟢 안전" if prob<30 else "🟠 주의")
            c3.metric("지능 레벨", intel)
            
            if silent: st.error("🔇 침묵의 메아리: 자극적 내용이나 근거 없음")
            if check_official(uploader): st.success(f"🛡️ 공식 언론사({uploader})")
            
            st.divider()
            c1,c2 = st.columns([1,1])
            with c1:
                st.info(f"🎯 쿼리: {query}")
                st.write("**영상 요약**"); st.caption(summarize(full_text))
                st.table(pd.DataFrame([["기본",50],["벡터",v_score],["뉴스",n_score],["페널티",silent+mismatch],["태그오용",tag_abuse_score]], columns=["항목","점수"]))
            with c2:
                colored_bar("진실", ts, "green"); colored_bar("거짓", fs, "red")
                st.write(f"**뉴스 ({news_cnt}건)**"); st.table(news_res) if news_res else st.warning("뉴스 없음")
                st.write("**여론**"); st.caption(f"{rel_msg} (논란어 {red_cnt}회)")
                
        except Exception as e: st.error(f"분석 중 오류: {e}")

# --- [App] ---
st.title("⚖️ Triple-Evidence Intelligence Forensic v48.3")
url = st.text_input("🔗 유튜브 URL")
if st.button("🚀 분석 시작") and url: run_main(url)

st.divider()
st.subheader("🗂️ 학습 데이터 (Cloud)")
try:
    df = pd.DataFrame(supabase.table("analysis_history").select("*").order("id", desc=True).execute().data)
    if not df.empty:
        if st.session_state["is_admin"]:
            ed = st.data_editor(df, column_config={"Delete":st.column_config.CheckboxColumn(default=False)}, disabled=["id","video_title"], hide_index=True)
            if "Delete" in ed.columns and st.button("삭제"):
                for i, r in ed[ed.Delete].iterrows(): supabase.table("analysis_history").delete().eq("id", r['id']).execute()
                st.success("삭제됨"); st.rerun()
        else: st.dataframe(df, hide_index=True)
except: pass
