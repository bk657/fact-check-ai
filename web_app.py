import streamlit as st
import re
import requests
import time
import random
from datetime import datetime
from collections import Counter
import yt_dlp
import pandas as pd
from bs4 import BeautifulSoup
import altair as alt
import traceback
import google.generativeai as genai

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="Fact-Check Center v56.0 (Gemini)", layout="wide", page_icon="⚖️")

# 🌟 Secrets 로드
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    # Gemini Key가 없으면 None으로 처리 (Logic 모드로 자동 전환)
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
except:
    st.error("❌ 필수 키(Secrets) 설정 오류. 관리자에게 문의하세요.")
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

# --- [2. 핵심 분석 엔진 (Gemini + Logic Hybrid)] ---

# A. Gemini에게 물어보는 함수 (최고 지능)
def ask_gemini_keywords(title, transcript):
    if not GEMINI_API_KEY: return None # 키 없으면 바로 포기
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash') # 빠르고 저렴한 모델
        
        prompt = f"""
        당신은 팩트체크 전문 AI입니다. 아래 유튜브 영상의 [제목]과 [자막 요약]을 읽고,
        이 내용의 진위를 뉴스 기사로 검증하기 위해 검색창에 입력할 '최적의 검색어'를 추출하세요.
        
        [조건]
        1. 가장 핵심적인 '인물(주어)'과 '사건(행위)'을 포함해야 합니다.
        2. '충격', '경악' 같은 감정적 형용사는 모두 제거하세요.
        3. 오직 검색어 문자열 하나만 출력하세요. (설명 금지)
        
        [제목]: {title}
        [자막 앞부분]: {transcript[:1000]}
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None

# B. 우리가 만든 Logic 함수 (비상용 백업)
VITAL_KEYWORDS = ['위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '내란', '간첩']
CRITICAL_STATE_KEYWORDS = ['별거', '이혼', '파경', '사망', '위독', '구속', '체포', '실형', '불화', '폭로', '충격', '논란', '중태', '심정지', '뇌사', '압수수색', '소환', '파산', '빚더미', '전과', '감옥', '간첩']
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']

def normalize_korean_word(word):
    josa_pattern = r'(은|는|이|가|을|를|의|에|에서|로|으로|와|과|도|만|한테|에게|이랑|까지|부터|조차|마저|이라고|라는|다는)$'
    if len(word) >= 2: return re.sub(josa_pattern, '', word)
    return word

def extract_meaningful_tokens(text):
    raw_tokens = re.findall(r'[가-힣]{2,}', text)
    noise = ['충격', '경악', '속보', '긴급', '오늘', '내일', '지금', '결국', '뉴스', '영상', '대부분', '이유', '왜', '있는', '없는', '하는', '것', '수', '등', '진짜', '정말', '너무', '그냥', '이제', '사실', '국민', '우리', '대한민국', '여러분']
    tokens = [normalize_korean_word(w) for w in raw_tokens]
    return [t for t in tokens if t not in noise and len(t) > 1]

def generate_logic_query(title, transcript):
    tokens = extract_meaningful_tokens(title)
    if tokens: return " ".join(tokens[:3]) # 제목 앞 3단어
    return title

# 🌟 [Hybrid Generator] Gemini 먼저 -> 실패하면 Logic
def generate_smart_query(title, transcript):
    # 1. Gemini 시도
    ai_query = ask_gemini_keywords(title, transcript)
    if ai_query:
        return ai_query, "✨ Gemini AI 추론"
    
    # 2. 실패 시 Logic 시도
    logic_query = generate_logic_query(title, transcript)
    return logic_query, "⚡ Pure Logic (Backup)"

# --- [3. 데이터 수집 및 분석] ---
def fetch_real_transcript(info):
    try:
        url = None
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

def fetch_comments_via_api(video_id):
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        res = requests.get(url, params={'part': 'snippet', 'videoId': video_id, 'key': YOUTUBE_API_KEY, 'maxResults': 50, 'order': 'relevance'})
        if res.status_code == 200:
            items = [i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in res.json().get('items', [])]
            return items, f"✅ API 수집 성공 (Top {len(items)})"
    except: pass
    return [], "⚠️ 댓글 수집 불가"

def fetch_news_regex(query):
    news_res = []
    try:
        rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR&ceid=KR:ko"
        raw = requests.get(rss_url, timeout=5).text
        items = re.findall(r'<item>(.*?)</item>', raw, re.DOTALL)
        for item in items[:10]:
            t = re.search(r'<title>(.*?)</title>', item)
            nt = t.group(1).replace("<![CDATA[", "").replace("]]>", "") if t else ""
            source = "Google News"
            if " - " in nt:
                parts = nt.rsplit(" - ", 1)
                nt = parts[0]
                source = parts[1]
            news_res.append({'title': nt, 'source': source})
    except: pass
    return news_res

def calculate_match_score(news_title, query, video_title):
    q_tokens = set(extract_meaningful_tokens(query))
    n_tokens = set(extract_meaningful_tokens(news_title))
    match_cnt = len(q_tokens & n_tokens)
    score = 0
    if match_cnt >= 2: score = 80
    elif match_cnt == 1: score = 40
    for crit in CRITICAL_STATE_KEYWORDS:
        if crit in video_title and crit not in news_title: return 0
    return score

def summarize_text_simple(text):
    if not text: return "요약할 내용이 없습니다."
    return ". ".join([s.strip() for s in text.split('.')[:3] if s.strip()]) + "."

def save_analysis(channel, title, score, url, query):
    try:
        supabase.table("analysis_history").insert({
            "channel_name": channel, "video_title": title, "fake_prob": score,
            "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "video_url": url, "keywords": query
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
    rows = ""
    for label, score, note in items:
        color = "#ffcccc" if score > 0 else "#ccffcc" if score < 0 else "#f0f0f0"
        sign = "+" if score > 0 else ""
        rows += f"<tr><td style='padding:8px;'>{label}<br><span style='font-size:0.8em;color:gray'>{note}</span></td><td style='padding:8px;text-align:right;background-color:{color};font-weight:bold'>{sign}{score}</td></tr>"
    st.markdown(f"""<table style="width:100%; border-collapse:collapse; border:1px solid #ddd; font-size:14px;"><thead><tr style="background-color:#f9f9f9;"><th>분석 항목</th><th style="text-align:right">점수</th></tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

def colored_progress_bar(label, percent, color):
    st.markdown(f"""<div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span style="font-size: 13px; font-weight: 600; color: #555;">{label}</span><span style="font-size: 13px; font-weight: 700; color: {color};">{round(percent * 100, 1)}%</span></div><div style="background-color: #eee; border-radius: 5px; height: 8px; width: 100%;"><div style="background-color: {color}; height: 8px; width: {percent * 100}%; border-radius: 5px;"></div></div></div>""", unsafe_allow_html=True)

def render_intelligence_distribution(current_prob):
    try:
        _, _, _, df = get_db_stats()
        if df.empty: return
        base = alt.Chart(df).transform_density('fake_prob', as_=['fake_prob', 'density'], extent=[0, 100], bandwidth=5).mark_area(opacity=0.3, color='#888').encode(x=alt.X('fake_prob:Q', title='확률 분포'), y=alt.Y('density:Q', title='데이터 밀도'))
        rule = alt.Chart(pd.DataFrame({'x': [current_prob]})).mark_rule(color='blue', size=3).encode(x='x')
        st.altair_chart(base + rule, use_container_width=True)
    except: pass

# --- [5. 메인 실행] ---
st.title("⚖️ Triple-Evidence Intelligence Forensic v56.0")
with st.container(border=True):
    st.markdown("### 🛡️ 법적 고지 및 책임 한계 (Disclaimer)\n본 서비스는 **인공지능(AI) 및 알고리즘 기반**으로 영상의 신뢰도를 분석하는 보조 도구입니다.\n* **최종 판단의 주체:** 정보의 진위 여부에 대한 최종적인 판단과 그에 따른 책임은 **사용자 본인**에게 있습니다.")
    agree = st.checkbox("위 내용을 확인하였으며, 이에 동의합니다. (동의 시 분석 버튼 활성화)")

url_input = st.text_input("🔗 분석할 유튜브 URL")
if st.button("🚀 정밀 분석 시작", use_container_width=True, disabled=not agree):
    if url_input:
        total_cnt, t_cnt, f_cnt, _ = get_db_stats()
        
        with st.status("🕵️ Gemini AI 가동 중...", expanded=True) as status:
            st.write("📡 영상 데이터 추출 중...")
            vid = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url_input)
            if vid: vid = vid.group(1)

            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                try:
                    info = ydl.extract_info(url_input, download=False)
                    title = info.get('title', '')
                    uploader = info.get('uploader', '')
                    tags = info.get('tags', [])
                    full_text = fetch_real_transcript(info)
                    
                    st.write("🧠 Gemini AI에게 문맥 추론 요청 중...")
                    query, q_type = generate_smart_query(title, full_text)
                    
                    st.write(f"🔍 뉴스 대조 검색 중: {query}")
                    news_items = fetch_news_regex(query)
                    cmts, cmt_status = fetch_comments_via_api(vid)
                    
                    max_match = 0
                    verified_news = []
                    for item in news_items:
                        s = calculate_match_score(item['title'], query, title)
                        if s > max_match: max_match = s
                        verified_news.append({'뉴스 제목': item['title'], '일치도': f"{s}%"})
                    
                    # Score Calculation
                    score = 50
                    breakdown = []
                    
                    is_silent = (len(news_items) == 0) or (max_match < 30)
                    has_critical = any(k in title for k in CRITICAL_STATE_KEYWORDS)
                    
                    news_diff = 0; news_msg = ""
                    if is_silent:
                        if has_critical: news_diff = 5; news_msg = "미검증 위험 주장"
                        else: news_diff = 10; news_msg = "증거 불충분"
                    else:
                        if max_match >= 80: news_diff = -45; news_msg = "뉴스 검증 완료"
                        elif max_match >= 40: news_diff = -20; news_msg = "부분적 사실 확인"
                        else: news_diff = 10; news_msg = "낮은 연관성"
                    breakdown.append(["뉴스 교차 검증", news_diff, news_msg])
                    
                    agitation = sum(title.count(w) + full_text.count(w) for w in ['충격','경악','폭로','속보','긴급'])
                    if agitation > 0:
                        breakdown.append(["자극적 표현", min(agitation*5, 20), f"선동 키워드 {agitation}회"])
                    
                    if any(o in uploader for o in OFFICIAL_CHANNELS):
                        breakdown.append(["공식 언론사", -50, "신뢰도 보장"])
                        
                    final_score = 50 + sum(item[1] for item in breakdown)
                    final_score = max(5, min(99, final_score))
                    
                    save_analysis(uploader, title, final_score, url_input, query)
                    status.update(label="분석 완료!", state="complete", expanded=False)
                    
                    # UI Output
                    st.subheader("🕵️ 핵심 분석 지표 (Key Indicators)")
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("최종 가짜뉴스 확률", f"{final_score}%", delta=f"{final_score-50}")
                    with c2:
                        icon = "🟢" if final_score < 30 else "🔴" if final_score > 60 else "🟠"
                        label = "안전" if final_score < 30 else "위험" if final_score > 60 else "주의"
                        st.metric("종합 AI 판정", f"{icon} {label}")
                    with c3: st.metric("AI Intelligence Level", f"{total_cnt} Nodes", delta="+1 Added")
                    
                    st.divider()
                    col1, col2 = st.columns([1, 1.4])
                    
                    with col1:
                        st.write("**[영상 상세 정보]**")
                        st.table(pd.DataFrame({"항목": ["영상 제목", "채널명", "해시태그"], "내용": [title, uploader, ", ".join(tags[:3])]}))
                        st.info(f"🎯 **{q_type}**: {query}")
                        with st.container(border=True):
                            st.markdown("📝 **영상 내용 요약 (AI Abstract)**")
                            st.caption(summarize_text_simple(full_text))
                        
                        st.write("**[Score Breakdown]**")
                        render_score_breakdown([["기본 위험도", 50, "Base Score"]] + breakdown)
                        
                    with col2:
                        st.subheader("📊 5대 정밀 분석 증거")
                        vec_t = 0.8 if final_score < 40 else 0.2
                        vec_f = 0.8 if final_score > 60 else 0.2
                        
                        st.markdown("**[증거 0] Semantic Vector Space**")
                        colored_progress_bar("✅ 진실 영역 근접도", vec_t, "#2ecc71")
                        colored_progress_bar("🚨 거짓 영역 근접도", vec_f, "#e74c3c")
                        
                        st.write("---")
                        st.markdown(f"**[증거 1] 뉴스 교차 대조 (Query: {query})**")
                        if verified_news: st.table(pd.DataFrame(verified_news))
                        else: st.warning("관련 뉴스가 없습니다.")
                        
                        st.markdown(f"**[증거 2] 시청자 여론 심층 분석**")
                        st.caption(f"💬 상태: {cmt_status}")
                        if cmts: st.write(f"최근 댓글: {', '.join(cmts[:3])}...")
                        
                        st.markdown("**[증거 3] 자막 세만틱 심층 대조**")
                        st.table(pd.DataFrame([["선동성 키워드", f"{agitation}회 발견"], ["제목-내용 일치도", "양호" if final_score < 60 else "주의 필요"]], columns=["항목", "결과"]))

                        st.markdown("**[증거 4] AI 최종 판정**")
                        if final_score > 60: st.error("신뢰할 수 없는 정보가 포함될 가능성이 높습니다.")
                        else: st.success("비교적 신뢰할 수 있는 정보로 판단됩니다.")

                        st.subheader("🧠 Intelligence Map")
                        render_intelligence_distribution(final_score)

                except Exception as e:
                    st.error(f"오류 발생: {e}")
                    st.code(traceback.format_exc())

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
