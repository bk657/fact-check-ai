import streamlit as st
import google.generativeai as genai
import re
import requests
import time
import json
import yt_dlp
import pandas as pd
import altair as alt
from datetime import datetime

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="Fact-Check Center v62.0 (Final)", layout="wide", page_icon="⚖️")

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

# 🌟 서비스 초기화 (검증된 gemini-pro 연결)
@st.cache_resource
def init_services():
    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # 🚨 성공한 모델: gemini-pro
        model = genai.GenerativeModel('gemini-pro')
        return sb, model
    except Exception as e:
        return None, None

supabase, gemini_model = init_services()

# --- [2. Gemini AI 에이전트] ---
class GeminiAgent:
    def __init__(self, model):
        self.model = model

    def extract_keywords(self, title, transcript):
        """뉴스 검색을 위한 최적의 키워드 추출"""
        if not self.model: return title
        prompt = f"""
        Extract the single most important search query to fact-check this video.
        - Input: {title}
        - Context: {transcript[:500]}
        - Output: ONLY the keyword string (e.g., 'Jay Lee Divorce')
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except: return title

    def analyze_content(self, title, channel, transcript, news_context, comments):
        """뉴스, 자막, 댓글을 종합하여 팩트체크 수행"""
        if not self.model:
            return {"fake_prob": 50, "verdict": "오류", "summary": "AI 연결 끊김", "clickbait_score": 0}

        prompt = f"""
        You are a professional Fact-Checker. Analyze the video claims against the news facts.
        Respond in JSON format ONLY.

        [Input Data]
        - Video Title: {title}
        - Video Transcript: {transcript[:4000]}
        - Related News: {news_context}
        - User Comments: {comments}

        [Tasks]
        1. Compare Video Claims vs News Facts.
        2. If News matches claims -> Low fake_prob (0-30).
        3. If News contradicts or No News -> High fake_prob (70-100).
        4. Translate all output values to Korean.

        [JSON Output Format]
        {{
            "summary": "3 bullet points summary in Korean",
            "fake_prob": Integer (0-100),
            "verdict": "위험/주의/안전",
            "reasoning": "Detailed reasoning in Korean (citing news results)",
            "fact_check_status": "Short status in Korean (e.g., '뉴스 교차 검증 완료')",
            "clickbait_score": Integer (0-100)
        }}
        """
        try:
            response = self.model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            return {
                "summary": "분석 실패", "fake_prob": 50, "verdict": "오류",
                "reasoning": "데이터 처리 중 문제가 발생했습니다.",
                "fact_check_status": "분석 불가", "clickbait_score": 0
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
    st.header("🛡️ 관리자 메뉴")
    if not st.session_state.get("is_admin"):
        if st.button("로그인"):
            st.session_state["is_admin"] = True
            st.rerun()
    else:
        st.success("관리자 로그인됨")
        if st.button("로그아웃"):
            st.session_state["is_admin"] = False
            st.rerun()

st.title("⚖️ Fact-Check Center v62.0")
st.caption("Powered by Google Gemini Pro")

with st.container(border=True):
    st.info("💡 **Gemini AI**가 영상 내용을 분석하고, 실시간 뉴스와 대조하여 진위 여부를 판별합니다.")
    url_input = st.text_input("분석할 유튜브 URL을 입력하세요")
    
    if st.button("🚀 정밀 분석 시작", type="primary", use_container_width=True):
        if url_input:
            if not gemini_model:
                st.error("⚠️ 시스템 초기화 오류: 잠시 후 다시 시도해주세요.")
            else:
                with st.status("🕵️ Gemini AI가 분석 중입니다...", expanded=True) as status:
                    
                    st.write("📥 영상 데이터(자막/메타정보) 다운로드...")
                    v_info = fetch_youtube_info(url_input)
                    if not v_info:
                        st.error("영상 정보를 가져오지 못했습니다. URL을 확인해주세요.")
                        st.stop()
                    
                    st.write("🧠 문맥 분석 및 검색 키워드 추출 중...")
                    search_keyword = gemini_agent.extract_keywords(v_info['title'], v_info['transcript'])
                    st.info(f"👉 생성된 검색어: **{search_keyword}**")
                    
                    st.write("📰 관련 뉴스 기사 및 팩트 탐색 중...")
                    news_result = fetch_google_news(search_keyword)
                    
                    st.write("⚖️ 주장 vs 사실 교차 검증 수행 중...")
                    comments = fetch_comments(v_info['id'])
                    result = gemini_agent.analyze_content(
                        v_info['title'], v_info['channel'], v_info['transcript'], news_result, comments
                    )
                    
                    save_data = {**v_info, **result}
                    save_history(save_data)
                    
                    status.update(label="✅ 분석이 완료되었습니다!", state="complete", expanded=False)

                # --- 결과 리포트 ---
                st.divider()
                
                # 1. 상단 핵심 지표
                c1, c2, c3 = st.columns(3)
                c1.metric("가짜뉴스 위험도", f"{result['fake_prob']}%", delta="High Risk" if result['fake_prob']>50 else "-Safe")
                c2.metric("AI 최종 판정", result['verdict'])
                c3.metric("낚시성 지수", f"{result['clickbait_score']}점")
                
                # 2. 상세 판정 이유
                if result['fake_prob'] > 60:
                    st.error(f"🚨 **위험 감지**: {result['reasoning']}")
                elif result['fake_prob'] < 40:
                    st.success(f"✅ **안전**: {result['reasoning']}")
                else:
                    st.warning(f"⚠️ **주의**: {result['reasoning']}")
                    
                # 3. 상세 분석 내용
                col_l, col_r = st.columns([1.2, 1])
                with col_l:
                    st.subheader("📝 AI 분석 리포트")
                    st.caption(f"검증 상태: {result['fact_check_status']}")
                    st.write(f"**핵심 요약**:\n{result['summary']}")
                    
                    with st.expander("🔍 참조된 뉴스 데이터 보기"):
                        st.write(news_result if news_result else "관련 기사 없음")

                with col_r:
                    st.subheader("📺 영상 정보")
                    st.table(pd.DataFrame({
                        "항목": ["채널명", "영상 제목", "자막 길이"],
                        "내용": [v_info['channel'], v_info['title'], f"{len(v_info['transcript']):,}자"]
                    }))

st.divider()
st.subheader("🗂️ 최근 분석 기록")
try:
    response = supabase.table("analysis_history").select("*").order("id", desc=True).limit(5).execute()
    if response.data:
        st.dataframe(pd.DataFrame(response.data)[['video_title', 'fake_prob', 'keywords', 'analysis_date']], use_container_width=True, hide_index=True)
except: pass
