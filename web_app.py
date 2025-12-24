import streamlit as st
import re
import requests
import time
import json
import yt_dlp
import pandas as pd
import altair as alt
from datetime import datetime
import google.generativeai as genai

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="Fact-Check v60.8 (Cache Fix)", layout="wide", page_icon="⚖️")

# 🌟 Secrets 로드
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError as e:
    st.error(f"❌ 필수 키 설정 누락: {e}")
    st.stop()

# 🌟 서비스 초기화 (캐시 버그 수정: API Key가 바뀌면 재실행)
@st.cache_resource
def init_services(api_key_signature): # 매개변수 추가로 캐시 리셋 유도
    sb = None
    model = None
    model_name = "None"
    
    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        genai.configure(api_key=api_key_signature) # 입력받은 키 사용
        
        # 연결 가능한 모델 자동 탐색
        candidates = ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.0-pro']
        for m in candidates:
            try:
                temp_model = genai.GenerativeModel(m)
                # 실제 통신 테스트
                if temp_model.generate_content("test"):
                    model = temp_model
                    model_name = m
                    break
            except: continue
            
    except Exception as e:
        return None, None, str(e)

    return sb, model, model_name

# 🚨 핵심: 키를 인자로 넘겨서 캐시를 갱신시킴
supabase, gemini_model, connected_model = init_services(GOOGLE_API_KEY)

# --- [2. Gemini AI 에이전트] ---
class GeminiAgent:
    def __init__(self, model):
        self.model = model

    def extract_keywords(self, title, transcript):
        if not self.model: return title
        prompt = f"""
        Extract ONE search keyword for fact-checking.
        Input: {title}
        Output: (Keyword Only)
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except: return title

    def analyze_content(self, title, channel, transcript, news_context, comments):
        if not self.model:
            return {"fake_prob": 50, "verdict": "오류", "summary": "AI 연결 실패", "clickbait_score": 0}

        prompt = f"""
        Analyze video claims vs news facts. Respond in JSON.

        [Data]
        - Title: {title}
        - Transcript: {transcript[:4000]}
        - News: {news_context}
        - Comments: {comments}

        [JSON Output]
        {{
            "summary": "Korean summary",
            "fake_prob": 0-100,
            "verdict": "위험/주의/안전",
            "reasoning": "Korean reasoning",
            "fact_check_status": "Status",
            "clickbait_score": 0-100
        }}
        """
        try:
            response = self.model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            return {
                "summary": "분석 실패",
                "fake_prob": 50,
                "verdict": "오류",
                "reasoning": f"에러: {str(e)}",
                "fact_check_status": "분석 불가",
                "clickbait_score": 0
            }

gemini_agent = GeminiAgent(gemini_model)

# --- [3. 유틸리티 함수] ---
def fetch_youtube_info(url):
    ydl_opts = {'quiet': True, 'skip_download': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            transcript = ""
            for sub_type in ['subtitles', 'automatic_captions']:
                if sub_type in info and 'ko' in info[sub_type]:
                    url = next((x['url'] for x in info[sub_type]['ko'] if x['ext'] == 'vtt'), None)
                    if url: 
                        transcript = requests.get(url).text
                        break
            
            clean_text = ""
            if transcript:
                lines = [line.strip() for line in transcript.splitlines() if '-->' not in line and line.strip() and not line.startswith(('WEBVTT', 'NOTE'))]
                clean_text = " ".join(list(dict.fromkeys(lines)))
            else:
                clean_text = info.get('description', '')

            return {
                "id": info['id'], "title": info.get('title', ''), 
                "channel": info.get('uploader', ''), "transcript": clean_text
            }
        except: return None

def fetch_google_news(query):
    try:
        rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, timeout=5)
        items = re.findall(r'<item>(.*?)</item>', res.text, re.DOTALL)
        news_list = []
        for item in items[:5]:
            t = re.search(r'<title>(.*?)</title>', item)
            if t: news_list.append(t.group(1).replace("<![CDATA[", "").replace("]]>", ""))
        return " | ".join(news_list) if news_list else "관련 뉴스 기사 없음"
    except: return "뉴스 검색 실패"

def fetch_comments(video_id):
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {'part': 'snippet', 'videoId': video_id, 'key': YOUTUBE_API_KEY, 'maxResults': 10, 'order': 'relevance'}
        res = requests.get(url, params=params)
        if res.status_code == 200:
            return " | ".join([i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in res.json().get('items', [])])
    except: pass
    return "댓글 없음"

def save_history(data):
    try:
        supabase.table("analysis_history").insert({
            "channel_name": data['channel'], "video_title": data['title'],
            "fake_prob": data['fake_prob'], "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "video_url": f"https://youtu.be/{data['id']}", "keywords": data['fact_check_status']
        }).execute()
    except: pass

# --- [4. UI 구성] ---
with st.sidebar:
    st.header("🛡️ 관리자")
    # 🌟 연결 상태 확인
    if connected_model and connected_model != "None":
        st.success(f"✅ AI Connected: {connected_model}")
    else:
        st.error(f"❌ Connection Failed: {connected_model}")
    
    if not st.session_state.get("is_admin"):
        if st.button("Login"):
            st.session_state["is_admin"] = True
            st.rerun()

st.title("⚖️ Fact-Check Center v60.8")
st.caption("Gemini Cache-Fix Engine")

with st.container(border=True):
    url_input = st.text_input("유튜브 URL 입력")
    if st.button("🚀 분석 시작", type="primary", use_container_width=True):
        if url_input:
            if not gemini_model:
                st.error("⚠️ AI 모델 연결에 실패했습니다. (캐시 리셋 시도됨)")
            else:
                with st.status(f"🕵️ Gemini ({connected_model}) 분석 중...", expanded=True) as status:
                    
                    st.write("📥 영상 데이터 추출 중...")
                    v_info = fetch_youtube_info(url_input)
                    if not v_info:
                        st.error("영상 정보를 가져오지 못했습니다.")
                        st.stop()
                    
                    st.write("🧠 Gemini: 뉴스 검색용 핵심 키워드 추출 중...")
                    search_keyword = gemini_agent.extract_keywords(v_info['title'], v_info['transcript'])
                    st.info(f"👉 추출된 검색어: **{search_keyword}**")
                    
                    st.write(f"📰 '{search_keyword}' 관련 뉴스 검색 중...")
                    news_result = fetch_google_news(search_keyword)
                    
                    st.write("⚖️ 팩트 교차 검증 및 판결 중...")
                    comments = fetch_comments(v_info['id'])
                    result = gemini_agent.analyze_content(
                        v_info['title'], v_info['channel'], v_info['transcript'], news_result, comments
                    )
                    
                    save_data = {**v_info, **result}
                    save_history(save_data)
                    
                    status.update(label="✅ 분석 완료!", state="complete", expanded=False)

                # --- 결과 표시 ---
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("가짜뉴스 위험도", f"{result['fake_prob']}%", delta="High" if result['fake_prob']>50 else "-Safe")
                c2.metric("AI 판정", result['verdict'])
                c3.metric("낚시성 지수", f"{result['clickbait_score']}점")
                
                if result['fake_prob'] > 60:
                    st.error(f"🚨 **주의**: {result['reasoning']}")
                else:
                    st.success(f"✅ **양호**: {result['reasoning']}")
                    
                st.subheader("📝 상세 분석 리포트")
                st.info(f"**검증 상태**: {result['fact_check_status']}")
                st.write(f"**요약**: {result['summary']}")
                
                with st.expander("🔍 참조된 뉴스 기사 데이터"):
                    st.write(news_result)

st.divider()
st.subheader("🗂️ 분석 기록")
try:
    response = supabase.table("analysis_history").select("*").order("id", desc=True).limit(5).execute()
    if response.data:
        st.dataframe(pd.DataFrame(response.data)[['video_title', 'fake_prob', 'keywords', 'analysis_date']])
except: pass
