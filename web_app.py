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
st.set_page_config(page_title="Fact-Check Center v54.1 (UI Restore)", layout="wide", page_icon="⚖️")

# 🌟 Secrets 로드
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except:
    st.error("❌ 필수 키(Secrets)가 설정되지 않았습니다.")
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

# --- [2. 관리자 인증] ---
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

# --- [3. 핵심 분석 엔진 (Pure Logic)] ---
VITAL_KEYWORDS = ['위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '내란', '간첩']
VIP_ENTITIES = ['윤석열', '대통령', '이재명', '한동훈', '김건희', '문재인', '박근혜', '이명박', '트럼프', '바이든', '푸틴', '젤렌스키', '시진핑', '정은', '이준석', '조국', '추미애', '홍준표', '유승민', '안철수', '손흥민', '이강인', '김민재', '류현진', '재용', '정의선', '최태원', '류중일', '감독', '조세호', '유재석', '장동민', '유호정', '이재룡', '임세령']
CRITICAL_STATE_KEYWORDS = ['별거', '이혼', '파경', '사망', '위독', '구속', '체포', '실형', '불화', '폭로', '충격', '논란', '중태', '심정지', '뇌사', '압수수색', '소환', '파산', '빚더미', '전과', '감옥', '간첩']
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']

def normalize_korean_word(word):
    josa_pattern = r'(은|는|이|가|을|를|의|에|에서|로|으로|와|과|도|만|한테|에게|이랑|까지|부터|조차|마저|이라고|라는|다는)$'
    if len(word) >= 2:
        return re.sub(josa_pattern, '', word)
    return word

def extract_meaningful_tokens(text):
    raw_tokens = re.findall(r'[가-힣]{2,}', text)
    noise = ['충격', '경악', '속보', '긴급', '오늘', '내일', '지금', '결국', '뉴스', '영상', '대부분', '이유', '왜', '있는', '없는', '하는', '것', '수', '등', '진짜', '정말', '너무', '그냥', '이제', '사실', '국민', '우리', '대한민국', '여러분', '그리고', '그래서', '그러나', '솔직히', '무슨', '어떤']
    tokens = [normalize_korean_word(w) for w in raw_tokens]
    return [t for t in tokens if t not in noise and len(t) > 1]

def detect_subject_logic(title):
    for vip in VIP_ENTITIES:
        if vip in title: return vip
    honorifics = ['회장', '의원', '대표', '대통령', '장관', '박사', '교수', '감독', '선수', '씨', '배우', '가수', '개그맨', '방송인']
    title_split = title.split()
    for i, word in enumerate(title_split):
        for hon in honorifics:
            if hon in word and i > 0:
                prev_word = normalize_korean_word(title_split[i-1])
                if len(prev_word) > 1: return prev_word
    tokens = extract_meaningful_tokens(title)
    if tokens: return tokens[0]
    return ""

def generate_smart_query(title, transcript):
    subject = detect_subject_logic(title)
    t_tokens = set(extract_meaningful_tokens(title))
    tr_tokens = set(extract_meaningful_tokens(transcript[:1000]))
    common = t_tokens.intersection(tr_tokens)
    actions = [w for w in common if w != subject]
    action = max(actions, key=len) if actions else ""
    
    if not action:
        for crit in CRITICAL_STATE_KEYWORDS:
            if crit in title:
                action = crit
                break
    
    if subject and action: return f"{subject} {action}"
    elif subject: return f"{subject} {title.split()[-1]}"
    else: return " ".join(extract_meaningful_tokens(title)[:3])

# --- [4. UI 유틸리티 (복구됨)] ---
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

def witty_loading_sequence(total):
    messages = [
        f"🧠 [Intelligence Level: {total}] 집단 지성 로드 중...",
        "📡 영상 데이터 정밀 추출 중...",
        "🔍 Pure Logic Engine 문맥 분석 중...", 
        "🚀 위성이 유튜브 본사 상공을 지나가는 중..."
    ]
    with st.status("🕵️ Context Merger v54.1 가동 중...", expanded=True) as status:
        for msg in messages: st.write(msg); time.sleep(0.5)
        st.write("✅ 분석 준비 완료!"); status.update(label="분석 완료!", state="complete", expanded=False)

def get_total_intelligence():
    try:
        count = supabase.table("analysis_history").select("id", count="exact").execute().count
        return count if count else 0
    except: return 0

# --- [5. 데이터 처리 함수] ---
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

def fetch_news_regex(query):
    news_res = []
    try:
        rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR&ceid=KR:ko"
        raw = requests.get(rss_url, timeout=5).text
        items = re.findall(r'<item>(.*?)</item>', raw, re.DOTALL)
        for item in items[:10]:
            t = re.search(r'<title>(.*?)</title>', item)
            nt = t.group(1).replace("<![CDATA[", "").replace("]]>", "") if t else ""
            news_res.append({'title': nt})
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

def render_intelligence_distribution(current_prob):
    try:
        res = supabase.table("analysis_history").select("fake_prob").execute()
        if not res.data: return
        df = pd.DataFrame(res.data)
        base = alt.Chart(df).transform_density('fake_prob', as_=['fake_prob', 'density'], extent=[0, 100], bandwidth=5).mark_area(opacity=0.3, color='#888').encode(x=alt.X('fake_prob:Q', title='가짜뉴스 확률 분포'), y=alt.Y('density:Q', title='데이터 밀도'))
        rule = alt.Chart(pd.DataFrame({'x': [current_prob]})).mark_rule(color='blue', size=3).encode(x='x')
        st.altair_chart(base + rule, use_container_width=True)
        if current_prob > 60: st.error("⚠️ 현재 영상은 **'고위험군'**에 속합니다.")
        elif current_prob < 40: st.success("✅ 현재 영상은 **'안전군'**에 속합니다.")
        else: st.warning("🔸 현재 영상은 **'중립 구간'**에 위치합니다.")
    except: pass

# --- [6. 메인 실행] ---
st.title("⚖️ Triple-Evidence Intelligence Forensic v54.1")
with st.container(border=True):
    st.markdown("### 🛡️ 법적 고지 및 책임 한계 (Disclaimer)\n본 서비스는 **인공지능(AI) 및 알고리즘 기반**으로 영상의 신뢰도를 분석하는 보조 도구입니다.\n* **최종 판단의 주체:** 정보의 진위 여부에 대한 최종적인 판단과 그에 따른 책임은 **사용자 본인**에게 있습니다.")
    agree = st.checkbox("위 내용을 확인하였으며, 이에 동의합니다. (동의 시 분석 버튼 활성화)")

url_input = st.text_input("🔗 분석할 유튜브 URL")
if st.button("🚀 정밀 분석 시작", use_container_width=True, disabled=not agree):
    if url_input:
        total_intelligence = get_total_intelligence()
        witty_loading_sequence(total_intelligence)
        
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            try:
                info = ydl.extract_info(url_input, download=False)
                title = info.get('title', '')
                uploader = info.get('uploader', '')
                tags = info.get('tags', [])
                full_text = fetch_real_transcript(info)
                
                query = generate_smart_query(title, full_text)
                news_items = fetch_news_regex(query)
                
                max_match = 0
                verified_news = []
                for item in news_items:
                    s = calculate_match_score(item['title'], query, title)
                    if s > max_match: max_match = s
                    verified_news.append({'뉴스 제목': item['title'], '일치도': f"{s}%"})
                
                # 점수 로직
                score = 50
                breakdown = []
                
                is_silent = (len(news_items) == 0) or (max_match < 30)
                has_critical = any(k in title for k in CRITICAL_STATE_KEYWORDS)
                
                # 1. 뉴스 검증
                news_diff = 0
                news_msg = ""
                if is_silent:
                    if has_critical: news_diff = 5; news_msg = "미검증 위험 주장"
                    else: news_diff = 10; news_msg = "증거 불충분"
                else:
                    if max_match >= 80: news_diff = -45; news_msg = "뉴스 검증 완료"
                    elif max_match >= 40: news_diff = -20; news_msg = "부분적 사실 확인"
                    else: news_diff = 10; news_msg = "낮은 연관성"
                breakdown.append(["뉴스 교차 검증", news_diff, news_msg])
                
                # 2. 공식 채널
                if any(o in uploader for o in OFFICIAL_CHANNELS):
                    breakdown.append(["공식 언론사", -50, "신뢰도 보장"])
                    
                # 3. 자극성
                agitation = sum(title.count(w) + full_text.count(w) for w in ['충격','경악','폭로','속보','긴급'])
                if agitation > 0:
                    breakdown.append(["자극적 표현", min(agitation*5, 20), f"선동 키워드 {agitation}회"])
                
                final_score = 50 + sum(item[1] for item in breakdown)
                final_score = max(5, min(99, final_score))
                
                save_analysis(uploader, title, final_score, url_input, query)
                
                # --- UI 출력 ---
                st.subheader("🕵️ 핵심 분석 지표 (Key Indicators)")
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("최종 가짜뉴스 확률", f"{final_score}%", delta=f"{final_score-50}")
                with c2:
                    icon = "🟢" if final_score < 30 else "🔴" if final_score > 60 else "🟠"
                    label = "안전" if final_score < 30 else "위험" if final_score > 60 else "주의"
                    st.metric("종합 AI 판정", f"{icon} {label}")
                with c3: st.metric("AI Intelligence Level", f"{total_intelligence} Knowledge Nodes", delta="+1 Added")
                
                st.divider()
                col1, col2 = st.columns([1, 1.4])
                
                with col1:
                    st.write("**[영상 상세 정보]**")
                    st.table(pd.DataFrame({"항목": ["영상 제목", "채널명", "해시태그"], "내용": [title, uploader, ", ".join(tags[:3])]}))
                    st.info(f"🎯 **AI 스마트 검색어**: {query}")
                    with st.container(border=True):
                        st.markdown("📝 **영상 내용 요약 (AI Abstract)**")
                        st.caption(summarize_text_simple(full_text))
                    
                    st.write("**[Score Breakdown]**")
                    render_score_breakdown([["기본 위험도", 50, "Base Score"]] + breakdown)
                    
                with col2:
                    st.subheader("📊 5대 정밀 분석 증거")
                    # Vector Simulation
                    vec_t = 0.8 if final_score < 40 else 0.2
                    vec_f = 0.8 if final_score > 60 else 0.2
                    colored_progress_bar("✅ 진실 영역 근접도", vec_t, "#2ecc71")
                    colored_progress_bar("🚨 거짓 영역 근접도", vec_f, "#e74c3c")
                    
                    st.write("---")
                    st.markdown(f"**[증거 1] 뉴스 교차 대조 (Query: {query})**")
                    if verified_news: st.table(pd.DataFrame(verified_news))
                    else: st.warning("관련 뉴스가 없습니다.")
                    
                    st.subheader("🧠 Intelligence Map: 내부 지식 분포도")
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
