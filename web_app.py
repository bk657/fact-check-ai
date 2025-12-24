import streamlit as st
from supabase import create_client
import google.generativeai as genai
import re
import requests
import time
import json
import yt_dlp
import pandas as pd
import altair as alt
from datetime import datetime
from collections import Counter

# --- [1. 시스템 설정 및 초기화] ---
st.set_page_config(page_title="Fact-Check Center v60.0 (Gemini Core)", layout="wide", page_icon="⚖️")

# 🌟 Secrets 로드
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] # Gemini 키 추가 필수
except KeyError as e:
    st.error(f"❌ 필수 키 설정 누락: {e}")
    st.stop()

# 🌟 서비스 초기화
@st.cache_resource
def init_services():
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash') # 속도와 가성비 최적화
    return sb, model

supabase, gemini_model = init_services()

# --- [2. Gemini AI 에이전트 클래스] ---
class GeminiAgent:
    def __init__(self, model):
        self.model = model

    def analyze_content(self, title, channel, transcript, news_context, comments):
        """
        Gemini에게 영상 내용, 뉴스 검색 결과, 댓글 반응을 종합적으로 분석 요청
        """
        prompt = f"""
        당신은 냉철하고 객관적인 '가짜뉴스 판별 전문 AI'입니다. 아래 제공된 데이터를 바탕으로 영상을 분석하여 JSON 포맷으로 응답하세요.

        [분석 대상]
        - 영상 제목: {title}
        - 채널명: {channel}
        - 자막(내용): {transcript[:15000]} (너무 길면 잘림)
        - 관련 뉴스 검색 결과: {news_context}
        - 시청자 댓글 반응: {comments}

        [분석 지침]
        1. **요약**: 영상의 핵심 주장 3가지를 요약하세요.
        2. **팩트체크**: 영상의 주장이 뉴스 검색 결과(Facts)와 일치하는지 교차 검증하세요. 뉴스 결과가 없거나 주장을 뒷받침하지 못하면 가짜 확률을 높이세요.
        3. **선동성 판단**: 제목이나 내용에 과도한 감정적 언어(충격, 경악 등)나 근거 없는 루머가 있는지 판단하세요.
        4. **최종 판정**: 0~100 사이의 '가짜뉴스/위험 확률(fake_prob)'을 산출하세요. (높을수록 위험)

        [출력 형식 (JSON)]
        {{
            "summary": "핵심 내용 3줄 요약",
            "fake_prob": 75,
            "verdict": "위험/주의/안전 중 택1",
            "reasoning": "점수 산정의 구체적인 이유 (200자 내외)",
            "fact_check_status": "뉴스 교차 검증 결과 (예: 근거 없음, 부분 일치, 확인 불가)",
            "clickbait_score": 0~100 (낚시성 점수)
        }}
        JSON 형식만 정확히 출력하세요. 마크다운 태그 없이 raw 텍스트로 주세요.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text.replace("```json", "").replace("```", ""))
        except Exception as e:
            return {"error": str(e), "fake_prob": 50, "summary": "AI 분석 실패", "reasoning": "API 오류 발생"}

gemini_agent = GeminiAgent(gemini_model)

# --- [3. 유틸리티 함수] ---
def fetch_youtube_info(url):
    """yt_dlp를 사용하여 영상 메타데이터와 자막 추출"""
    ydl_opts = {'quiet': True, 'skip_download': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            video_id = info['id']
            title = info.get('title', '')
            channel = info.get('uploader', '')
            
            # 자막 추출 로직
            transcript = ""
            if 'subtitles' in info and 'ko' in info['subtitles']:
                sub_url = next((x['url'] for x in info['subtitles']['ko'] if x['ext'] == 'vtt'), None)
                if sub_url:
                    res = requests.get(sub_url)
                    transcript = clean_vtt(res.text)
            
            # 자동 자막이라도 가져오기
            if not transcript and 'automatic_captions' in info and 'ko' in info['automatic_captions']:
                sub_url = next((x['url'] for x in info['automatic_captions']['ko'] if x['ext'] == 'vtt'), None)
                if sub_url:
                    res = requests.get(sub_url)
                    transcript = clean_vtt(res.text)
            
            if not transcript:
                transcript = info.get('description', '') # 자막 없으면 설명란 사용

            return {"id": video_id, "title": title, "channel": channel, "transcript": transcript}
        except Exception as e:
            return None

def clean_vtt(text):
    """VTT 자막 포맷 정리"""
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        if '-->' in line or line.strip() == '' or line.startswith('WEBVTT') or line.startswith('NOTE'):
            continue
        clean = re.sub(r'<[^>]+>', '', line).strip()
        if clean and clean not in clean_lines: # 중복 제거
            clean_lines.append(clean)
    return " ".join(clean_lines)

def fetch_google_news(query):
    """구글 뉴스 RSS 검색"""
    try:
        rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, timeout=5)
        items = re.findall(r'<item>(.*?)</item>', res.text, re.DOTALL)
        news_data = []
        for item in items[:5]: # 상위 5개만
            t = re.search(r'<title>(.*?)</title>', item)
            news_data.append(t.group(1).replace("<![CDATA[", "").replace("]]>", "") if t else "")
        return " | ".join(news_data) if news_data else "관련 뉴스 없음"
    except:
        return "뉴스 검색 실패"

def fetch_comments(video_id):
    """유튜브 API 댓글 수집"""
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {'part': 'snippet', 'videoId': video_id, 'key': YOUTUBE_API_KEY, 'maxResults': 20, 'order': 'relevance'}
        res = requests.get(url, params=params)
        if res.status_code == 200:
            comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in res.json().get('items', [])]
            return " | ".join(comments)
    except: pass
    return "댓글 수집 불가"

def save_history(data):
    try:
        supabase.table("analysis_history").insert({
            "channel_name": data['channel'],
            "video_title": data['title'],
            "fake_prob": data['fake_prob'],
            "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "video_url": f"https://youtu.be/{data['id']}",
            "keywords": data['verdict']
        }).execute()
    except Exception as e:
        print(f"DB Save Error: {e}")

# --- [4. UI 구성] ---
# 사이드바: 관리자
with st.sidebar:
    st.header("🛡️ 관리자 메뉴")
    if "is_admin" not in st.session_state: st.session_state["is_admin"] = False
    
    if not st.session_state["is_admin"]:
        pw = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if pw == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True
                st.rerun()
            else:
                st.error("비밀번호 불일치")
    else:
        st.success("Admin Logged In")
        if st.button("Logout"):
            st.session_state["is_admin"] = False
            st.rerun()

# 메인 UI
st.title("⚖️ Gemini Fact-Check Center v60.0")
st.caption("Powered by Google Gemini 1.5 & Streamlit")

with st.container(border=True):
    st.info("💡 **Google Gemini AI**가 영상 자막과 실시간 뉴스를 교차 검증하여 진위 여부를 판독합니다.")
    url_input = st.text_input("유튜브 영상 URL을 입력하세요")
    start_btn = st.button("🚀 AI 정밀 분석 시작", use_container_width=True, type="primary")

if start_btn and url_input:
    with st.status("🕵️ Gemini AI가 영상을 분석 중입니다...", expanded=True) as status:
        # 1. 영상 데이터 수집
        st.write("📥 영상 메타데이터 및 자막 추출 중...")
        video_info = fetch_youtube_info(url_input)
        
        if not video_info:
            status.update(label="영상 정보를 가져올 수 없습니다.", state="error")
            st.stop()
            
        # 2. 뉴스 및 댓글 데이터 수집
        st.write("📰 관련 뉴스 및 여론 데이터 수집 중...")
        # 검색어 최적화: 제목에서 특수문자 제거 후 사용
        clean_title = re.sub(r'[^\w\s]', '', video_info['title'])
        news_context = fetch_google_news(clean_title)
        comments = fetch_comments(video_info['id'])
        
        # 3. Gemini 분석 수행
        st.write("🧠 Gemini 1.5 모델 추론 및 팩트체크 수행 중...")
        ai_result = gemini_agent.analyze_content(
            video_info['title'],
            video_info['channel'],
            video_info['transcript'],
            news_context,
            comments
        )
        
        # 4. 저장
        save_data = {**video_info, **ai_result}
        save_history(save_data)
        
        status.update(label="✅ 분석 완료!", state="complete", expanded=False)

    # --- [결과 리포트] ---
    st.divider()
    
    # 상단 메트릭
    col1, col2, col3 = st.columns(3)
    prob = ai_result.get('fake_prob', 0)
    
    col1.metric("가짜뉴스 위험도", f"{prob}%", delta="High Risk" if prob > 60 else "-Safe")
    col2.metric("AI 판정", ai_result.get('verdict', '판단 불가'))
    col3.metric("낚시성 지수", f"{ai_result.get('clickbait_score', 0)}점")
    
    # 게이지 차트 (Altair)
    chart_df = pd.DataFrame({'value': [prob]})
    base = alt.Chart(chart_df).mark_bar().encode(x=alt.X('value', scale=alt.Scale(domain=[0, 100])))
    st.progress(prob / 100)
    
    if prob > 70:
        st.error(f"🚨 **위험 감지**: {ai_result.get('reasoning')}")
    elif prob < 30:
        st.success(f"✅ **안전**: {ai_result.get('reasoning')}")
    else:
        st.warning(f"⚠️ **주의**: {ai_result.get('reasoning')}")

    # 상세 내용
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.subheader("📝 AI 요약 & 분석")
        st.info(f"**요약**: {ai_result.get('summary')}")
        st.write(f"**팩트체크 상태**: {ai_result.get('fact_check_status')}")
        
        with st.expander("참조된 뉴스 데이터 보기"):
            st.write(news_context)

    with col_r:
        st.subheader("📺 영상 정보")
        st.table(pd.DataFrame({
            "항목": ["제목", "채널", "자막 길이"],
            "내용": [video_info['title'], video_info['channel'], f"{len(video_info['transcript']):,}자"]
        }))

# --- [5. 히스토리 (관리자 전용 기능 삭제 가능)] ---
st.divider()
st.subheader("🗂️ 최근 분석 기록")
try:
    rows = supabase.table("analysis_history").select("*").order("id", desc=True).limit(5).execute()
    if rows.data:
        df = pd.DataFrame(rows.data)
        st.dataframe(df[['video_title', 'fake_prob', 'analysis_date', 'keywords']], hide_index=True, use_container_width=True)
except:
    st.caption("데이터베이스 연결 대기 중...")
