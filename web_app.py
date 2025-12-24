import streamlit as st
import re
import requests
import time
import random
import google.generativeai as genai # 🌟 구글 AI 라이브러리
from datetime import datetime
from collections import Counter
import yt_dlp
import pandas as pd
from bs4 import BeautifulSoup
import altair as alt
import traceback

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="Fact-Check v57.0 (Gemini Powered)", layout="wide", page_icon="⚖️")

# 🌟 Secrets 로드 (Gemini Key 필수)
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] # 🌟 필수
except:
    st.error("❌ 필수 키(Secrets)가 설정되지 않았습니다. GEMINI_API_KEY를 확인하세요.")
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

# --- [2. 핵심 엔진: Gemini AI Keyword Extractor] ---
def ask_gemini_keywords(title, transcript):
    """
    Gemini에게 제목과 자막을 주고, 뉴스 검색용 최적 키워드를 받아옵니다.
    """
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash') # 빠르고 똑똑한 모델
        
        # 🌟 프롬프트 엔지니어링: AI에게 구체적인 지시를 내림
        prompt = f"""
        너는 팩트체크를 위한 전문 검색원이야. 
        아래 유튜브 영상의 [제목]과 [자막 요약]을 분석해서, 이 내용이 사실인지 뉴스 기사로 확인하기 위한 '최적의 검색어' 1개를 만들어줘.

        [조건]
        1. 영상의 핵심 주장(누가, 무엇을, 어떤 사건)이 포함되어야 해.
        2. '충격', '경악', '슬픈' 같은 감정적 수식어는 빼고, '팩트(명사)' 위주로 구성해.
        3. 예시: '이재용 회장의 슬픈 사연' (X) -> '이재용 이혼 사유' or '이재용 임세령 위자료' (O)
        4. 오직 검색어 문자열 하나만 출력해. (따옴표 없이)

        [제목]: {title}
        [자막 앞부분]: {transcript[:1500]}
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # 에러 나면 기존 로직(백업)으로 리턴
        print(f"Gemini Error: {e}")
        return None

# --- [3. 보조 기능 (기존 로직 유지)] ---
VITAL_KEYWORDS = ['위독', '사망', '별세', '구속', '체포', '기소', '실형', '응급실', '이혼', '불화', '파경', '충격', '경악', '속보', '긴급', '폭로', '양성', '확진', '심정지', '뇌사', '중태', '압수수색', '소환', '퇴진', '탄핵', '내란', '간첩']
CRITICAL_STATE_KEYWORDS = ['별거', '이혼', '파경', '사망', '위독', '구속', '체포', '실형', '불화', '폭로', '충격', '논란', '중태', '심정지', '뇌사', '압수수색', '소환', '파산', '빚더미', '전과', '감옥', '간첩']
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']

def normalize_korean_word(word):
    word = re.sub(r'[^가-힣0-9]', '', word)
    josa_list = ['은', '는', '이', '가', '을', '를', '의', '에', '에게', '로', '으로', '와', '과', '도', '만', '한테', '까지', '부터']
    for josa in josa_list:
        if word.endswith(josa): return word[:-len(josa)]
    return word

def extract_meaningful_tokens(text):
    raw_tokens = re.findall(r'[가-힣]{2,}', text)
    noise = ['충격', '경악', '속보', '긴급', '오늘', '내일', '지금', '결국', '뉴스', '영상', '대부분', '이유', '왜', '있는', '없는', '하는', '것', '수', '등', '진짜', '정말', '너무', '그냥', '이제', '사실', '국민', '우리', '대한민국', '여러분']
    return [normalize_korean_word(w) for w in raw_tokens if normalize_korean_word(w) not in noise]

def generate_backup_query(title):
    # Gemini 실패 시 사용할 백업 로직
    tokens = extract_meaningful_tokens(title)
    return " ".join(tokens[:3]) if tokens else title

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
            if res.status_code == 200:
                content = res.text
                if "#EXTM3U" in content: return None
                clean = []
                for line in content.splitlines():
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
            return items
    except: pass
    return []

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

def calculate_match_score(news_title, query):
    q_tokens = set(extract_meaningful_tokens(query))
    n_tokens = set(extract_meaningful_tokens(news_title))
    match_cnt = len(q_tokens & n_tokens)
    if match_cnt >= 2: return 80
    elif match_cnt == 1: return 40
    return 0

def summarize_text_simple(text):
    return ". ".join([s.strip() for s in text.split('.')[:3] if s.strip()]) + "."

def save_analysis(channel, title, score, url, query):
    try:
        supabase.table("analysis_history").insert({
            "channel_name": channel, "video_title": title, "fake_prob": score,
            "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "video_url": url, "keywords": query
        }).execute()
    except: pass

def render_score_breakdown(data_list):
    style = """<style>table.score-table { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; font-family: sans-serif; font-size: 14px; margin-top: 10px;} table.score-table th { background-color: #f8f9fa; color: #495057; font-weight: bold; padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; } table.score-table td { padding: 12px 15px; border-bottom: 1px solid #f0f0f0; color: #333; } table.score-table tr:last-child td { border-bottom: none; } .badge { padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-align: center; min-width: 45px; } .badge-danger { background-color: #ffebee; color: #d32f2f; } .badge-success { background-color: #e8f5e9; color: #2e7d32; } .badge-neutral { background-color: #f5f5f5; color: #757575; border: 1px solid #e0e0e0; }</style>"""
    rows = ""
    for item, score, note in data_list:
        badge = f'<span class="badge badge-danger">+{score}</span>' if score > 0 else f'<span class="badge badge-success">{score}</span>' if score < 0 else f'<span class="badge badge-neutral">0</span>'
        rows += f"<tr><td>{item}<br><span style='color:#888; font-size:11px;'>{note}</span></td><td style='text-align: right;'>{badge}</td></tr>"
    st.markdown(f"{style}<table class='score-table'><thead><tr><th>분석 항목</th><th style='text-align: right;'>점수</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def colored_progress_bar(label, percent, color):
    st.markdown(f"""<div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span style="font-size: 13px; font-weight: 600; color: #555;">{label}</span><span style="font-size: 13px; font-weight: 700; color: {color};">{round(percent * 100, 1)}%</span></div><div style="background-color: #eee; border-radius: 5px; height: 8px; width: 100%;"><div style="background-color: {color}; height: 8px; width: {percent * 100}%; border-radius: 5px;"></div></div></div>""", unsafe_allow_html=True)

# --- [4. 메인 실행] ---
st.title("⚖️ Fact-Check v57.0 (Gemini Engine)")
with st.container(border=True):
    st.markdown("### 🛡️ Disclaimer\n본 서비스는 **Gemini AI**를 활용하여 영상의 맥락을 분석하고 뉴스와 대조합니다. 최종 판단은 사용자에게 있습니다.")
    agree = st.checkbox("동의합니다.")

url_input = st.text_input("🔗 분석할 유튜브 URL")
if st.button("🚀 정밀 분석 시작", use_container_width=True, disabled=not agree):
    if url_input:
        vid = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url_input)
        if vid: vid = vid.group(1)

        with st.status("🕵️ Gemini AI가 영상을 분석 중입니다...", expanded=True) as status:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                try:
                    info = ydl.extract_info(url_input, download=False)
                    title = info.get('title', ''); uploader = info.get('uploader', '')
                    tags = info.get('tags', [])
                    
                    st.write("📝 자막(Transcript) 추출 중...")
                    full_text = fetch_real_transcript(info)
                    
                    # 🌟 [핵심] Gemini에게 키워드 추출 요청
                    st.write("🧠 Gemini가 최적의 검색어를 추론 중...")
                    query = ask_gemini_keywords(title, full_text)
                    
                    # Gemini가 실패하면 백업 로직 사용
                    q_source = "✨ Gemini AI"
                    if not query:
                        query = generate_backup_query(title)
                        q_source = "⚡ Backup Logic"
                    
                    st.write(f"🔍 뉴스 검색 실행: {query}")
                    news_items = fetch_news_regex(query)
                    cmts = fetch_comments_via_api(vid)
                    
                    # 분석
                    max_match = 0
                    verified_news = []
                    for item in news_items:
                        s = calculate_match_score(item['title'], query)
                        if s > max_match: max_match = s
                        verified_news.append({'뉴스 제목': item['title'], '일치도': f"{s}%"})
                    
                    # 점수 계산
                    score = 50
                    breakdown = []
                    
                    is_silent = (len(news_items) == 0) or (max_match < 30)
                    has_critical = any(k in title for k in CRITICAL_STATE_KEYWORDS)
                    
                    news_diff = 0; news_msg = ""
                    if is_silent:
                        if has_critical: news_diff = 5; news_msg = "미검증 위험 주장"
                        else: news_diff = 10; news_msg = "증거 불충분"
                    else:
                        if max_match >= 80: news_diff = -45; news_msg = "팩트 확인됨"
                        elif max_match >= 40: news_diff = -20; news_msg = "부분적 사실"
                        else: news_diff = 10; news_msg = "관련성 낮음"
                    breakdown.append(["뉴스 교차 검증", news_diff, news_msg])
                    
                    agitation = sum(title.count(w) + full_text.count(w) for w in ['충격','경악','폭로','속보','긴급'])
                    if agitation > 0:
                        breakdown.append(["자극적 표현", min(agitation*5, 20), f"선동 키워드 {agitation}회"])
                    
                    if any(o in uploader for o in OFFICIAL_CHANNELS):
                        breakdown.append(["공식 언론사", -50, "신뢰도 보장"])
                        
                    final_score = max(5, min(99, 50 + sum(item[1] for item in breakdown)))
                    save_analysis(uploader, title, final_score, url_input, query)
                    status.update(label="분석 완료!", state="complete", expanded=False)
                    
                    # --- UI 출력 ---
                    st.subheader("🕵️ 분석 결과")
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.metric("가짜뉴스 확률", f"{final_score}%", delta=f"{final_score-50}")
                        st.info(f"🎯 **{q_source} 추출 검색어**:\n{query}")
                        with st.expander("영상 요약 보기"):
                            st.write(summarize_text_simple(full_text))
                        st.caption("점수 상세:")
                        render_score_breakdown([["기본 위험도", 50, "Base Score"]] + breakdown)
                        
                    with c2:
                        st.subheader("📰 팩트체크 (뉴스 대조)")
                        if verified_news:
                            st.table(pd.DataFrame(verified_news))
                        else:
                            st.warning("관련된 뉴스 기사가 없습니다.")
                            
                        st.subheader("📊 정밀 지표")
                        colored_progress_bar("진실 근접도", 0.8 if final_score < 40 else 0.2, "#2ecc71")
                        colored_progress_bar("거짓 근접도", 0.8 if final_score > 60 else 0.2, "#e74c3c")
                        
                        if cmts:
                            st.markdown("**💬 시청자 반응 (최근 댓글)**")
                            st.write(", ".join(cmts[:3]) + "...")

                except Exception as e:
                    st.error(f"오류 발생: {e}")
                    st.code(traceback.format_exc())

st.divider()
st.subheader("🗂️ 분석 기록")
try:
    response = supabase.table("analysis_history").select("*").order("id", desc=True).execute()
    if response.data:
        st.dataframe(pd.DataFrame(response.data)[['video_title', 'fake_prob', 'keywords', 'analysis_date']])
except: pass
