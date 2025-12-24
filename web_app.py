import streamlit as st
import sys
import traceback

# --- [초안전 모드 설정] ---
st.set_page_config(page_title="Fact-Check v53.7 (Rescue)", layout="wide", page_icon="🛟")

# 에러 캡처 래퍼 (앱이 죽지 않게 보호)
def main_app():
    # 필수 라이브러리 임포트 (여기서 에러나면 잡힘)
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
    from bs4 import BeautifulSoup
    import altair as alt

    # 🌟 Secrets 확인
    try:
        YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
        ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    except Exception as e:
        st.error(f"❌ Secrets 설정 오류: {e}")
        st.stop()

    @st.cache_resource
    def init_supabase():
        return create_client(SUPABASE_URL, SUPABASE_KEY)

    try:
        supabase = init_supabase()
    except Exception as e:
        st.error(f"DB 연결 실패: {e}")
        st.stop()

    # --- [상수 정의] ---
    WEIGHT_NEWS_DEFAULT = 45; WEIGHT_VECTOR = 35
    PENALTY_ABUSE = 20; PENALTY_NO_FACT = 25; PENALTY_SILENT_ECHO = 40
    
    VIP_ENTITIES = ['윤석열', '대통령', '이재명', '한동훈', '김건희', '문재인', '박근혜', '이명박', '트럼프', '바이든', '푸틴', '젤렌스키', '시진핑', '정은', '이준석', '조국', '추미애', '홍준표', '유승민', '안철수', '손흥민', '이강인', '김민재', '류현진', '재용', '정의선', '최태원', '류중일', '감독', '조세호', '유재석', '장동민', '유호정', '이재룡', '임세령']
    CRITICAL_STATE_KEYWORDS = ['별거', '이혼', '파경', '사망', '위독', '구속', '체포', '실형', '불화', '폭로', '충격', '논란', '중태', '심정지', '뇌사', '압수수색', '소환', '파산', '빚더미', '전과', '감옥', '간첩']
    OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']

    # --- [핵심 로직: Pure Python NLP] ---
    def normalize_korean_word(word):
        # 조사 제거 (Regex)
        josa_pattern = r'(은|는|이|가|을|를|의|에|에서|로|으로|와|과|도|만|한테|에게|이랑|까지|부터|조차|마저)$'
        if len(word) >= 2:
            return re.sub(josa_pattern, '', word)
        return word

    def extract_meaningful_tokens(text):
        raw_tokens = re.findall(r'[가-힣]{2,}', text)
        noise = ['충격', '경악', '속보', '긴급', '오늘', '내일', '지금', '결국', '뉴스', '영상', '대부분', '이유', '왜', '있는', '없는', '하는', '것', '수', '등', '진짜', '정말', '너무', '그냥', '이제', '사실', '국민', '우리', '대한민국', '여러분']
        return [normalize_korean_word(w) for w in raw_tokens if normalize_korean_word(w) not in noise]

    # 🌟 [v53.7] Logic-based Subject Detector
    def detect_subject_pure_logic(title, text):
        # 1. VIP 리스트 매칭
        for vip in VIP_ENTITIES:
            if vip in title: return vip
        
        # 2. 호칭 기반 추론
        honorifics = ['회장', '의원', '대표', '대통령', '장관', '박사', '교수', '감독', '선수', '씨', '배우', '가수', '개그맨']
        words = title.split()
        for i, word in enumerate(words):
            for hon in honorifics:
                if hon in word and i > 0:
                    return normalize_korean_word(words[i-1])
        return ""

    def extract_action_pure_logic(title, transcript):
        t_tokens = set(extract_meaningful_tokens(title))
        tr_tokens = extract_meaningful_tokens(transcript[:1000])
        common = t_tokens.intersection(tr_tokens)
        common = [w for w in common if w not in VIP_ENTITIES]
        if common: return max(common, key=len)
        return ""

    def generate_smart_query(title, transcript):
        subject = detect_subject_pure_logic(title, transcript)
        action = extract_action_pure_logic(title, transcript)
        
        if not subject:
            tokens = extract_meaningful_tokens(title)
            subject = tokens[0] if tokens else ""
            
        final_query = f"{subject} {action}".strip()
        if len(final_query) < 3:
            final_query = " ".join(extract_meaningful_tokens(title)[:3])
        return final_query

    # --- [Data Functions] ---
    def save_analysis(channel, title, prob, url, keywords):
        try: supabase.table("analysis_history").insert({"channel_name": channel, "video_title": title, "fake_prob": prob, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url": url, "keywords": keywords}).execute()
        except: pass

    def get_db_stats():
        try:
            res = supabase.table("analysis_history").select("fake_prob").execute()
            if res.data:
                df = pd.DataFrame(res.data)
                return len(df), len(df[df['fake_prob'] < 40]), len(df[df['fake_prob'] > 60]), df
        except: pass
        return 0, 0, 0, pd.DataFrame()

    # --- [Utils] ---
    def summarize_transcript(text, title):
        if not text or len(text) < 50: return "⚠️ 내용 부족"
        clean = re.sub(r'http\S+|#EXTM3U|#EXT-X-VERSION:3|\[.*?\]|[>]+', '', text)
        sentences = re.split(r'(?<=[.?!])\s+', clean)
        if len(sentences) <= 3: return clean.strip()
        
        # 간단 요약 로직
        title_tokens = set(extract_meaningful_tokens(title))
        scored = []
        for i, s in enumerate(sentences):
            if len(s) < 15: continue
            score = sum(1 for w in extract_meaningful_tokens(s) if w in title_tokens) * 5
            if i < len(sentences)*0.2: score += 2
            scored.append((i, s, score))
        
        top = sorted(scored, key=lambda x:x[2], reverse=True)[:3]
        top.sort(key=lambda x:x[0])
        return " ".join([s[1] for s in top])

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
            rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
            raw = requests.get(rss, timeout=5).text
            items = re.findall(r'<item>(.*?)</item>', raw, re.DOTALL)
            for item in items[:10]:
                t = re.search(r'<title>(.*?)</title>', item)
                nt = t.group(1).replace("<![CDATA[", "").replace("]]>", "") if t else ""
                news_res.append({'title': nt})
        except: pass
        return news_res

    def calculate_match(news_item, query_str, transcript):
        # 1. 뉴스 제목과 쿼리 일치도
        t_tokens = set(extract_meaningful_tokens(news_item['title']))
        q_tokens = set(extract_meaningful_tokens(query_str))
        score = 100 if len(q_tokens & t_tokens) >= 2 else 50 if len(q_tokens & t_tokens) >= 1 else 0
        
        # 2. Critical Check
        for crit in CRITICAL_STATE_KEYWORDS:
            if crit in query_str and crit not in news_item['title']:
                return 0
        return score

    # --- [UI] ---
    with st.sidebar:
        st.header("🛡️ 관리자 메뉴")
        if st.session_state.get("is_admin", False):
            st.success("✅ 로그인됨")
            if st.button("로그아웃"): st.session_state["is_admin"] = False; st.rerun()
        else:
            with st.form("login"):
                if st.form_submit_button("로그인"):
                    if st.text_input("PW", type="password") == ADMIN_PASSWORD:
                        st.session_state["is_admin"] = True; st.rerun()

    st.title("⚖️ Fact-Check Center v53.7 (Rescue)")
    
    total, t_cnt, f_cnt, df_stats = get_db_stats()
    
    url = st.text_input("🔗 분석할 유튜브 URL")
    if st.button("🚀 정밀 분석 시작") and url:
        with st.status("🕵️ 분석 중...", expanded=True) as status:
            st.write("📡 영상 정보 추출 중...")
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', '')
                uploader = info.get('uploader', '')
                tags = info.get('tags', [])
                full_text = fetch_real_transcript(info)
            
            st.write("🧠 스마트 쿼리 생성 중...")
            query = generate_smart_query(title, full_text)
            
            st.write(f"🔎 뉴스 검색 중: {query}")
            news_items = fetch_news_regex(query)
            
            max_match = 0
            verified_news = []
            for item in news_items:
                m = calculate_match(item, query, full_text)
                if m > max_match: max_match = m
                verified_news.append({'뉴스 제목': item['title'], '일치도': f"{m}%"})
            
            # 점수 산정
            is_silent = (len(news_items) == 0) or (max_match < 20)
            has_critical = any(k in title for k in CRITICAL_STATE_KEYWORDS)
            agitation = sum(title.count(w) + full_text.count(w) for w in ['충격','경악','폭로','속보'])
            
            score = 50 # Base
            
            note = ""
            if is_silent:
                if has_critical:
                    score += 5; note = "⚠️ 미검증 위험 주장 (+5)"
                elif agitation >= 3:
                    score += 40; note = "🔇 침묵의 메아리 (+40)"
            else:
                if max_match >= 60: score -= 45; note = "✅ 뉴스 검증 완료 (-45)"
                else: score += 15; note = "⚠️ 낮은 일치도 (+15)"
            
            if any(o in uploader.upper() for o in OFFICIAL_CHANNELS):
                score = 5; note = "🛡️ 공식 언론사"

            save_analysis(uploader, title, score, url, query)
            status.update(label="분석 완료!", state="complete")

        # 결과 표시
        col1, col2 = st.columns(2)
        with col1:
            st.metric("가짜뉴스 확률", f"{score}%", delta=note)
            st.info(f"🎯 검색어: {query}")
            st.caption(summarize_transcript(full_text, title))
            
        with col2:
            st.subheader("뉴스 대조 결과")
            if verified_news: st.table(pd.DataFrame(verified_news))
            else: st.warning("관련 뉴스가 없습니다.")
            
            if not df_stats.empty:
                st.subheader("DB 분포도")
                c = alt.Chart(df_stats).transform_density('fake_prob', as_=['fake_prob', 'density'], extent=[0, 100]).mark_area(opacity=0.3).encode(x='fake_prob:Q', y='density:Q')
                rule = alt.Chart(pd.DataFrame({'x': [score]})).mark_rule(color='red').encode(x='x')
                st.altair_chart(c + rule, use_container_width=True)

# 실행 진입점 (Crash Catch)
if __name__ == "__main__":
    try:
        main_app()
    except Exception as e:
        st.error("🚨 치명적 오류 발생 (앱 보호 모드)")
        st.code(traceback.format_exc())
        st.info("관리자에게 위 에러 코드를 전달해주세요.")
