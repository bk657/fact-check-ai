import streamlit as st
import sys
import subprocess

# --- [라이브러리 버전 확인] ---
try:
    import google.generativeai as genai
    lib_version = genai.__version__
except ImportError:
    lib_version = "Not Installed"

st.set_page_config(page_title="Fact-Check v61.1 (Auto-Negotiation)", layout="wide", page_icon="⚖️")

# --- [시스템 설정] ---
import re
import requests
import time
import json
import yt_dlp
import pandas as pd
import altair as alt
from datetime import datetime

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

# 🌟 서비스 초기화 (자동 모델 협상 로직)
@st.cache_resource
def init_services():
    sb = None
    model = None
    connected_name = "None"
    
    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # 🚨 [핵심] 연결 가능한 모델을 순서대로 테스트
        # 1.5-flash가 404면 gemini-pro로, 그것도 안되면 1.0으로 자동 넘어감
        candidates = [
            'gemini-1.5-flash',
            'gemini-pro',       # 가장 안정적
            'gemini-1.5-pro',
            'gemini-1.0-pro'
        ]
        
        for m_name in candidates:
            try:
                temp_model = genai.GenerativeModel(m_name)
                # 실제 통신 테스트 (Ping)
                response = temp_model.generate_content("Hi")
                if response:
                    model = temp_model
                    connected_name = m_name
                    break # 성공하면 루프 종료
            except Exception:
                continue # 실패하면 다음 후보 시도

    except Exception as e:
        return None, None, str(e)

    return sb, model, connected_name

supabase, gemini_model, model_name = init_services()

# --- [Gemini AI 에이전트] ---
class GeminiAgent:
    def __init__(self, model):
        self.model = model

    def extract_keywords(self, title, transcript):
        if not self.model: return title
        prompt = f"Extract ONE search keyword for: {title}. Context: {transcript[:500]}"
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except: return title

    def analyze_content(self, title, channel, transcript, news_context, comments):
        if not self.model:
            return {"fake_prob": 50, "verdict": "시스템 오류", "summary": "AI 연결 실패", "clickbait_score": 0}

        prompt = f"""
        Analyze logic. Respond JSON.
        Data: {title}, {news_context}, {transcript[:3000]}, {comments}
        
        JSON Format:
        {{
            "summary": "Korean summary text",
            "fake_prob": 0-100,
            "verdict": "Text",
            "reasoning": "Korean text",
            "fact_check_status": "Korean text",
            "clickbait_score": 0-100
        }}
        """
        try:
            response = self.model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            return {"summary": "Error", "fake_prob": 50, "verdict": "Error", "reasoning": str(e), "fact_check_status": "Error", "clickbait_score": 0}

gemini_agent = GeminiAgent(gemini_model)

# --- [유틸리티 함수] ---
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

# --- [UI 구성] ---
with st.sidebar:
    st.header("🛡️ 관리자")
    st.success(f"Lib: v{lib_version}")
    
    # 🌟 연결된 모델 확인 (성공 시 모델명이 뜸)
    if model_name and model_name != "None":
        st.success(f"✅ Active: {model_name}")
    else:
        st.error("❌ No Model Available")
        
    if not st.session_state.get("is_admin"):
        if st.button("Login"):
            st.session_state["is_admin"] = True
            st.rerun()

st.title("⚖️ Fact-Check Center v61.1")
st.caption("Gemini Auto-Negotiation Engine")

with st.container(border=True):
    url_input = st.text_input("유튜브 URL 입력")
    if st.button("🚀 분석 시작", type="primary", use_container_width=True):
        if url_input:
            if not gemini_model:
                st.error("⚠️ 사용 가능한 AI 모델을 찾을 수 없습니다. (API Key 문제 가능성)")
            else:
                with st.status(f"🕵️ Gemini ({model_name}) 분석 중...", expanded=True) as status:
                    
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
