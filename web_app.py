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

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="Fact-Check Center v60.1 (Keyword Logic Fix)", layout="wide", page_icon="⚖️")

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

# 🌟 서비스 초기화
@st.cache_resource
def init_services():
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return sb, model
    except Exception as e:
        st.error(f"초기화 실패: {e}")
        return None, None

supabase, gemini_model = init_services()

# --- [2. Gemini AI 에이전트 (2단계 로직)] ---
class GeminiAgent:
    def __init__(self, model):
        self.model = model

    def extract_keywords(self, title, transcript):
        """
        1단계: 뉴스 검색을 위한 '최적의 키워드' 추출
        """
        prompt = f"""
        너는 팩트체크 검색원이야. 아래 유튜브 영상 내용을 확인하고, 실제 뉴스 기사를 찾기 위한 '검색용 키워드'를 1개만 추출해.
        
        [조건]
        1. 자극적인 형용사(충격, 경악 등)는 모두 제거해.
        2. '인물명'과 '핵심 사건(명사)' 위주로 조합해.
        3. 예시: '이재용의 충격적인 눈물' -> '이재용 눈물 이유'
        4. 오직 키워드 문자열만 출력해. (설명 금지)

        제목: {title}
        내용: {transcript[:1000]}
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return title # 실패하면 제목 그대로 사용

    def analyze_content(self, title, channel, transcript, news_context, comments):
        """
        2단계: 수집된 정보를 바탕으로 최종 분석 (JSON 출력)
        """
        prompt = f"""
        당신은 팩트체크 전문 AI입니다. 아래 데이터를 분석하여 JSON 형식으로 응답하세요.

        [데이터]
        - 영상 제목: {title}
        - 뉴스 검색 결과: {news_context}
        - 영상 자막: {transcript[:10000]}
        - 댓글 여론: {comments}

        [지시사항]
        1. **fake_prob**: 뉴스 검색 결과와 영상 주장이 다르면 점수를 높게(80~100), 일치하면 낮게(0~30) 책정하세요. 뉴스가 아예 없으면 '근거 없음'으로 간주하여 60~80점을 주세요.
        2. **verdict**: 점수에 따라 '위험', '주의', '안전' 중 하나 선택.
        3. **fact_check_status**: 뉴스 기사와 대조했을 때의 결과를 한 문장으로 요약. (예: "관련 보도 확인됨", "근거 없는 루머")

        [출력 포맷 (JSON)]
        {{
            "summary": "영상 핵심 내용 3줄 요약",
            "fake_prob": 0~100 숫자,
            "verdict": "위험/주의/안전",
            "reasoning": "판단 이유 (뉴스 대조 결과 포함)",
            "fact_check_status": "팩트체크 상태 요약",
            "clickbait_score": 0~100 숫자
        }}
        """
        try:
            response = self.model.generate_content(prompt)
            # JSON 파싱 안전장치
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            return {
                "summary": "AI 분석 중 오류가 발생했습니다.",
                "fake_prob": 50,
                "verdict": "오류",
                "reasoning": f"데이터 처리 실패: {str(e)}",
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
            # 자막 추출 시도
            if 'subtitles' in info and 'ko' in info['subtitles']:
                url = next((x['url'] for x in info['subtitles']['ko'] if x['ext'] == 'vtt'), None)
                if url: transcript = requests.get(url).text
            elif 'automatic_captions' in info and 'ko' in info['automatic_captions']:
                url = next((x['url'] for x in info['automatic_captions']['ko'] if x['ext'] == 'vtt'), None)
                if url: transcript = requests.get(url).text
            
            # VTT 클리닝
            clean_text = ""
            if transcript:
                lines = [line.strip() for line in transcript.splitlines() if '-->' not in line and line.strip() and not line.startswith(('WEBVTT', 'NOTE'))]
                clean_text = " ".join(list(dict.fromkeys(lines))) # 중복 제거
            else:
                clean_text = info.get('description', '')

            return {
                "id": info['id'], "title": info.get('title', ''), 
                "channel": info.get('uploader', ''), "transcript": clean_text
            }
        except: return None

def fetch_google_news(query):
    try:
        # 정확도를 위해 쿼리 인코딩
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
    if not st.session_state.get("is_admin"):
        if st.button("Login"):
            st.session_state["is_admin"] = True
            st.rerun()

st.title("⚖️ Fact-Check Center v60.1")
st.caption("Gemini AI Based • Keyword Optimization Engine")

with st.container(border=True):
    url_input = st.text_input("유튜브 URL 입력")
    if st.button("🚀 분석 시작", type="primary", use_container_width=True):
        if url_input and gemini_model:
            with st.status("🕵️ AI 분석 프로세스 가동...", expanded=True) as status:
                
                # 1. 영상 정보
                st.write("📥 영상 데이터 추출 중...")
                v_info = fetch_youtube_info(url_input)
                if not v_info:
                    st.error("영상 정보를 가져오지 못했습니다.")
                    st.stop()
                
                # 2. 키워드 추출 (핵심!)
                st.write("🧠 Gemini: 뉴스 검색용 핵심 키워드 추출 중...")
                search_keyword = gemini_agent.extract_keywords(v_info['title'], v_info['transcript'])
                st.info(f"👉 추출된 검색어: **{search_keyword}**")
                
                # 3. 뉴스 검색
                st.write(f"📰 '{search_keyword}' 관련 뉴스 검색 중...")
                news_result = fetch_google_news(search_keyword)
                
                # 4. 종합 분석
                st.write("⚖️ 팩트 교차 검증 및 판결 중...")
                comments = fetch_comments(v_info['id'])
                result = gemini_agent.analyze_content(
                    v_info['title'], v_info['channel'], v_info['transcript'], news_result, comments
                )
                
                # 저장
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
