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
st.set_page_config(page_title="Fact-Check Center v60.6 (Auto-Fix)", layout="wide", page_icon="⚖️")

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

# 🌟 서비스 초기화 (자동 모델 탐색 로직 탑재)
@st.cache_resource
def init_services():
    sb = None
    model = None
    selected_model_name = "Unknown"
    
    try:
        # DB 연결
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Gemini 설정
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # 🚨 [핵심] 사용 가능한 모델 목록 조회 및 자동 선택
        available_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        except:
            pass

        # 우선순위: 1.5-flash -> 1.5-pro -> 1.0-pro -> 아무거나
        if any('gemini-1.5-flash' in m for m in available_models):
            target_model = 'gemini-1.5-flash'
        elif any('gemini-1.5-pro' in m for m in available_models):
            target_model = 'gemini-1.5-pro'
        elif any('gemini-pro' in m for m in available_models):
            target_model = 'gemini-pro'
        elif available_models:
            target_model = available_models[0] # 뭐라도 있으면 그거 씀
        else:
            target_model = 'gemini-pro' # 목록 조회 실패시 기본값 강제 시도

        # 'models/' 접두사 제거 (라이브러리 호환성)
        if target_model.startswith('models/'):
            target_model = target_model.replace('models/', '')
            
        model = genai.GenerativeModel(target_model)
        selected_model_name = target_model

    except Exception as e:
        print(f"Init Error: {e}")
        return None, None, str(e)

    return sb, model, selected_model_name

supabase, gemini_model, model_name_log = init_services()

# --- [2. Gemini AI 에이전트] ---
class GeminiAgent:
    def __init__(self, model):
        self.model = model

    def extract_keywords(self, title, transcript):
        if not self.model: return title
        prompt = f"""
        Extract the one best search keyword for fact-checking.
        Input: {title}
        Context: {transcript[:500]}
        Output: (Keyword Only)
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return title

    def analyze_content(self, title, channel, transcript, news_context, comments):
        if not self.model:
            return {"fake_prob": 50, "verdict": "오류", "summary": "AI 연결 실패", "clickbait_score": 0}

        prompt = f"""
        Analyze this video claim against news facts. Respond in JSON.

        [Data]
        - Title: {title}
        - Transcript: {transcript[:4000]}
        - News Facts: {news_context}
        - Comments: {comments}

        [Output JSON]
        {{
            "summary": "Korean summary (3 lines)",
            "fake_prob": 0-100,
            "verdict": "위험/주의/안전",
            "reasoning": "Korean reasoning (Fact vs Claim)",
            "fact_check_status": "Verification result",
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
                "reasoning": f"에러 발생: {str(e)}",
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
    # 🌟 연결된 모델명 확인용 (디버깅)
    st.success(f"Connected Model: {model_name_log}")
    
    if not st.session_state.get("is_admin"):
        if st.button("Login"):
            st.session_state["is_admin"] = True
            st.rerun()

st.title("⚖️ Fact-Check Center v60.6")
st.caption("Gemini Auto-Selector Engine")

with st.container(border=True):
    url_input = st.text_input("유튜브 URL 입력")
    if st.button("🚀 분석 시작", type="primary", use_container_width=True):
        if url_input:
            if not gemini_model:
                st.error(f"⚠️ AI 모델 초기화 실패: {model_name_log}")
            else:
                with st.status("🕵️ Gemini AI 분석 중...", expanded=True) as status:
                    
                    st.write("📥 영상 데이터 추출 중...")
                    v_info = fetch_youtube_info(url_input)
                    if not v_info:
                        st.error("영상 정보를 가져오지 못했습니다.")
                        st.stop()
                    
                    st.write(f"🧠 Gemini({model_name_log}): 핵심 키워드 추출 중...")
                    search_keyword = gemini_agent.extract_keywords(v_info['title'], v_info['transcript'])
                    st.info(f"👉 추출된 검색어: **{search_keyword}**")
                    
                    st.write(f"📰 '{search_keyword}' 뉴스 검색 중...")
                    news_result = fetch_google_news(search_keyword)
                    
                    st.write("⚖️ 팩트 교차 검증 중...")
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
