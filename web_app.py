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
st.set_page_config(page_title="Fact-Check Center v47.1 (Final Fix)", layout="wide", page_icon="⚖️")

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
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

with st.sidebar:
    st.header("🛡️ 관리자 메뉴")
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
    else:
        st.info("데이터 삭제는 관리자만 가능합니다.")

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

# --- [UI Utils] ---
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
    nouns = re.findall(r'[가-힣]{2,}', text)
    return list(dict.fromkeys([n for n in nouns if n not in noise]))

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
                    if re.fullmatch(r'[가-힣A-Za-z0-9]+', prev_word):
                        if prev_word not in VITAL_KEYWORDS + ['충격', '속보']: prev_noun = prev_word
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

# --- [Main Execution] ---
def run_forensic_main(url):
    total_intelligence = train_dynamic_vector_engine()
    witty_loading_sequence(total_intelligence)
    
    vid = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if vid: vid = vid.group(1)

    with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', ''); uploader = info.get('uploader', '')
            tags = info.get('tags', []); desc = info.get('description', '')
            
            trans, t_status = fetch_real_transcript(info)
            full_text = trans if trans else desc
            
            is_official = check_is_official(uploader)
            
            # 🌟 [Fix] 변수명 통일 (is_ai_content -> is_ai_content)
            is_ai_content, ai_msg = detect_ai_content(info) 
            
            w_news = 70 if is_ai_content else WEIGHT_NEWS_DEFAULT
            w_vec = 10 if is_ai_content else WEIGHT_VECTOR
            
            query = generate_pinpoint_query(title, tags)
            hashtag_display = ", ".join([f"#{t}" for t in tags]) if tags else "해시태그 없음"
            abuse_score, abuse_msg = check_tag_abuse(title, tags, uploader)
            summary = summarize_transcript(full_text)
            agitation = count_sensational_words(full_text + title)
            
            ts, fs = vector_engine.analyze_position(query + " " + title)
            t_impact = int(ts * w_vec) * -1; f_impact = int(fs * w_vec)

            news_ev = []; max_match = 0
            try:
                rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
                r = requests.get(rss_url, timeout=5)
                root = ET.fromstring(r.content)
                items = root.findall('.//item')
                
                for item in items[:3]:
                    nt = item.find('title').text
                    d_tag = item.find('description')
                    nd = clean_html(d_tag.text) if d_tag is not None else ""
                    m = calculate_dual_match({'title': nt, 'desc': nd}, extract_nouns(query), full_text)
                    if m > max_match: max_match = m
                    news_ev.append({"뉴스 제목": nt, "최종 일치도": f"{m}%"})
            except: pass
            
            cmts, c_status = fetch_comments_via_api(vid)
            top_kw, rel_score, rel_msg = analyze_comment_relevance(cmts, title + " " + full_text)
            red_cnt, red_list = check_red_flags(cmts)
            is_controversial = red_cnt > 0
            
            w_news = 65 if is_controversial else w_news
            
            silent_penalty = 0; news_score = 0; mismatch_penalty = 0
            is_silent = (len(news_ev) == 0) or (len(news_ev) > 0 and max_match < 20)
            
            if is_silent:
                if agitation >= 3: silent_penalty = PENALTY_SILENT_ECHO; t_impact *= 2; f_impact *= 2
                else: mismatch_penalty = 10
            elif is_controversial:
                news_score = PENALTY_NO_FACT if max_match < 60 else int((max_match/100)**2 * w_news) * -1
            else:
                news_score = int((max_match/100)**2 * w_news) * -1
                
            if is_official: news_score = -50; mismatch_penalty = 0; silent_penalty = 0
            
            sent_score = 0
            if cmts and not is_controversial:
                neg = sum(1 for c in cmts for k in ['가짜','선동'] if k in c) / len(cmts)
                sent_score = int(neg * 10)
                
            clickbait = 10 if any(w in title for w in ['충격','경악','폭로']) else -5
            total = 50 + t_impact + f_impact + news_score + sent_score + clickbait + abuse_score + mismatch_penalty + silent_penalty
            prob = max(5, min(99, total))
            
            save_analysis(uploader, title, prob, url, query)

            # --- UI ---
            st.subheader("🕵️ 핵심 분석 지표 (Key Indicators)")
            col_a, col_b, col_c = st.columns(3)
            with col_a: st.metric("최종 가짜뉴스 확률", f"{prob}%", delta=f"{total - 50}")
            with col_b:
                icon = "🟢" if prob < 30 else "🔴" if prob > 60 else "🟠"
                verdict = "매우 안전" if prob < 30 else "위험 감지" if prob > 60 else "주의 요망"
                st.metric("종합 AI 판정", f"{icon} {verdict}")
            with col_c: st.metric("AI Intelligence Level", f"{total_intelligence} Knowledge Nodes", delta="+1 Added")

            if is_ai_content: st.warning(f"🤖 **AI 생성 콘텐츠 감지됨**: {ai_msg}")
            if is_official: st.success(f"🛡️ **공식 언론사 채널({uploader})입니다.**")
            if silent_penalty > 0: st.error("🔇 **침묵의 메아리(Silent Echo) 경고**: 근거 없는 자극적 주장")

            st.divider()
            col1, col2 = st.columns([1, 1.4])
            with col1:
                st.write("**[영상 상세 정보]**")
                st.table(pd.DataFrame({"항목": ["영상 제목", "채널명", "조회수", "해시태그"], "내용": [title, uploader, f"{info.get('view_count',0):,}회", hashtag_display]}))
                st.info(f"🎯 **핀포인트 뉴스 검색어**: {query}")
                with st.container(border=True):
                    st.markdown("📝 **영상 내용 요약 (AI Abstract)**")
                    st.caption("자막 데이터를 분석하여 핵심 문장 3개를 추출한 결과입니다.")
                    st.write(summary)
                st.write("**[Score Breakdown]**")
                render_score_breakdown([
                    ["기본 위험도", 50, "Base Score"],
                    ["진실 맥락 보너스 (벡터)", t_impact, ""], ["가짜 패턴 가점 (벡터)", f_impact, ""],
                    ["뉴스 교차 대조 (Dual)", news_safety_score, ""],
                    ["침묵의 메아리 (No News)", silent_penalty, ""],
                    ["여론/제목/자막 가감", sent_score + clickbait, ""],
                    ["내용 불일치 기만", mismatch_penalty, ""], ["해시태그 어뷰징", abuse_score, ""]
                ])

            with col2:
                st.subheader("📊 5대 정밀 분석 증거")
                st.markdown("**[증거 0] Semantic Vector Space (진실/거짓 분포)**")
                st.caption(f"💡 Intelligence Level {total_intelligence} 기반 분석")
                colored_progress_bar("✅ 진실 영역 근접도", ts, "#2ecc71")
                colored_progress_bar("🚨 거짓 영역 근접도", fs, "#e74c3c")
                st.write("---")
                st.markdown(f"**[증거 1] 뉴스 교차 대조 (Query: {query})**")
                st.caption(f"📡 수집: **{len(news_ev)}건**")
                if news_ev: st.table(pd.DataFrame(news_ev))
                else: st.warning("🔍 관련 뉴스를 찾을 수 없습니다. (Silent Echo Risk Increased)")
                st.markdown("**[증거 2] 시청자 여론 심층 분석**")
                st.caption(f"💬 상태: **{c_status}**")
                if cmts:
                    st.table(pd.DataFrame([["최다 빈출 키워드", ", ".join(top_kw)], ["논란 감지 여부", f"{red_cnt}회"], ["주제 일치도", f"{rel_score}% ({rel_msg})"]], columns=["항목", "내용"]))
                else: st.warning("⚠️ 댓글 수집 불가.")
                st.markdown("**[증거 3] 자막 세만틱 심층 대조**")
                st.caption(f"📝 **{t_status}** | 📚 전체 단어: **{len(full_text.split())}개**")
                st.table(pd.DataFrame([["제목 낚시어", "있음" if clickbait > 0 else "없음"], ["선동성 지수", f"{agitation}회"], ["기사-영상 일치도", f"{max_match}%"]], columns=["분석 항목", "판정 결과"]))
                st.markdown("**[증거 4] AI 최종 분석 판단**")
                st.success(f"🔍 현재 분석된 종합 점수는 {prob}점입니다.")
                if prob < 30 or prob > 70: st.toast(f"🤖 AI가 이 결과를 학습했습니다!", icon="🧠")

        except Exception as e: st.error(f"오류: {e}")

st.title("⚖️ Triple-Evidence Intelligence Forensic v47.1")
with st.container(border=True):
    st.markdown("### 🛡️ 법적 고지 및 책임 한계 (Disclaimer)\n본 서비스는 **인공지능(AI) 및 알고리즘 기반**으로 영상의 신뢰도를 분석하는 보조 도구입니다.\n* **최종 판단의 주체:** 정보의 진위 여부에 대한 최종적인 판단과 그에 따른 책임은 **사용자 본인**에게 있습니다.")
    agree = st.checkbox("위 내용을 확인하였으며, 이에 동의합니다. (동의 시 분석 버튼 활성화)")

url = st.text_input("🔗 분석할 유튜브 URL")
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
    df['Delete'] = False
    cols = ['Delete', 'id', 'analysis_date', 'video_title', 'fake_prob', 'keywords']
    df = df[cols]
    if st.session_state.get("is_admin", False):
        edited_df = st.data_editor(df, column_config={"Delete": st.column_config.CheckboxColumn("선택 삭제", default=False)}, disabled=["id", "analysis_date", "video_title", "keywords"], hide_index=True, use_container_width=True)
        to_delete = edited_df[edited_df.Delete]
        if not to_delete.empty:
            if st.button(f"🗑️ 선택한 {len(to_delete)}건의 기록 영구 삭제", type="primary"):
                try:
                    for index, row in to_delete.iterrows(): supabase.table("analysis_history").delete().eq("id", row['id']).execute()
                    st.success("✅ 삭제 완료!"); time.sleep(1); st.rerun()
                except Exception as e: st.error(f"삭제 중 오류 발생: {e}")
    else:
        st.dataframe(df.drop(columns=['Delete']), hide_index=True, use_container_width=True)
        st.info("🔒 데이터 삭제 권한이 없습니다. (관리자 로그인 필요)")
else: st.info("☁️ 클라우드 DB에 저장된 분석 기록이 없습니다.")
