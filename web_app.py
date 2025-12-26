import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from supabase import create_client
import google.generativeai as genai # 구글 라이브러리 사용

# -----------------------------------------------------------------------------
# 1. 설정 및 초기화 (Setup)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI 영상 분석기 (Gemini Ver)", layout="wide", page_icon="🎬")

# 비밀번호 및 API 키 로드
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    
    # [변경] OpenAI 대신 Google 키를 가져옵니다.
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    
    # 구글 Gemini 설정
    genai.configure(api_key=GOOGLE_API_KEY)
    
except Exception as e:
    st.error(f"❌ 설정 파일(Secrets) 로드 실패: {e}")
    st.info("secrets.toml 파일에 SUPABASE_URL, SUPABASE_KEY, ADMIN_PASSWORD, GOOGLE_API_KEY가 있는지 확인하세요.")
    st.stop()

# Supabase 클라이언트 연결
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 세션 상태 초기화
if "is_admin" not in st.session_state: st.session_state["is_admin"] = False
if "analysis_result" not in st.session_state: st.session_state["analysis_result"] = None

# -----------------------------------------------------------------------------
# 2. 유틸리티 클래스 & 함수 (Utils)
# -----------------------------------------------------------------------------

class VectorEngine:
    """구글 Gemini를 사용하여 텍스트를 벡터로 변환"""
    def get_embedding(self, text):
        try:
            # Gemini 임베딩 모델 사용 (text-embedding-004)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
                title="Embedding of text"
            )
            return result['embedding']
        except Exception as e:
            # 에러 발생 시 로그만 찍고 넘어감 (멈춤 방지)
            print(f"벡터 생성 실패: {e}")
            return None

vector_engine = VectorEngine()

def get_similar_content(current_keywords):
    """
    analysis_history 테이블에서 유사한 과거 영상을 찾습니다.
    """
    try:
        if not current_keywords: return []
        
        # 1. 현재 키워드 벡터 생성
        query_vector = vector_engine.get_embedding(current_keywords)
        if not query_vector: return []

        # 2. DB 조회 (analysis_history)
        response = supabase.table("analysis_history").select("video_title, video_url, vector_json").not_.is_("vector_json", "null").execute()
        
        candidates = []
        for row in response.data:
            if isinstance(row['vector_json'], str):
                vec = json.loads(row['vector_json'])
            else:
                vec = row['vector_json']
            
            if not vec: continue

            # 코사인 유사도 계산
            dot_product = np.dot(query_vector, vec)
            norm_a = np.linalg.norm(query_vector)
            norm_b = np.linalg.norm(vec)
            
            if norm_a == 0 or norm_b == 0: continue
            
            similarity = dot_product / (norm_a * norm_b)

            if similarity > 0.6: # 유사도 60% 이상
                candidates.append({
                    "title": row['video_title'],
                    "url": row['video_url'],
                    "score": similarity
                })
        
        return sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]
        
    except Exception as e:
        print(f"유사도 검색 에러: {e}")
        return []

def save_db(ch, ti, pr, url, kw, detail):
    """
    analysis_history 테이블에 저장
    """
    try: 
        # 벡터 생성
        embedding = vector_engine.get_embedding(kw + " " + ti)
        
        data_to_insert = {
            "channel_name": ch,
            "video_title": ti,
            "fake_prob": pr,
            "video_url": url, 
            "keywords": kw,
            "detail_json": detail,
            "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "vector_json": embedding
        }
        
        supabase.table("analysis_history").insert(data_to_insert).execute()
        st.toast("✅ DB 저장 및 학습 완료!", icon="💾")
        time.sleep(1)
        
    except Exception as e: 
        st.error(f"❌ 데이터베이스 저장 실패: {e}")

# -----------------------------------------------------------------------------
# 3. 메인 UI
# -----------------------------------------------------------------------------
st.title("🎬 AI 유튜브 분석기 (Gemini Powered)")

# 사이드바
with st.sidebar:
    st.header("설정")
    if not st.session_state["is_admin"]:
        pwd = st.text_input("관리자 비밀번호", type="password")
        if st.button("로그인"):
            if pwd == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True
                st.rerun()
            else:
                st.error("비밀번호 불일치")
    else:
        st.success("관리자 로그인 됨")
        if st.button("로그아웃"):
            st.session_state["is_admin"] = False
            st.rerun()

# URL 입력
url_input = st.text_input("유튜브 URL 입력", placeholder="https://youtu.be/...")

if st.button("🚀 분석 시작", type="primary"):
    if not url_input:
        st.warning("URL을 입력해주세요.")
    else:
        try:
            # 중복 체크 (analysis_history)
            check = supabase.table("analysis_history").select("*").eq("video_url", url_input).execute()
            if check.data:
                st.info("💡 이미 분석된 데이터입니다.")
                res = check.data[0]
                st.session_state["analysis_result"] = {
                    "channel": res['channel_name'],
                    "title": res['video_title'],
                    "prob": res['fake_prob'],
                    "keywords": res['keywords'],
                    "detail": res['detail_json'],
                    "url": res['video_url']
                }
            else:
                with st.spinner("AI 분석 중... (Gemini)"):
                    time.sleep(1.5)
                    # --- 실제 AI 분석 로직이 들어갈 곳 ---
                    result_data = {
                        "channel": "분석된 채널",
                        "title": "영상 제목 예시",
                        "prob": 85,
                        "keywords": "Gemini, AI, 테스트",
                        "detail": {"summary": "Gemini 분석 결과입니다."},
                        "url": url_input
                    }
                    st.session_state["analysis_result"] = result_data
                    
        except Exception as e:
            st.error(f"분석 오류: {e}")

# 결과 화면
if st.session_state["analysis_result"]:
    res = st.session_state["analysis_result"]
    
    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("🚫 조작 의심 확률", f"{res['prob']}%")
        if res['prob'] > 70: st.error("⚠️ 주의")
        else: st.success("✅ 안전")
    
    with c2:
        st.subheader(res['title'])
        st.write(f"📺 채널: **{res['channel']}**")
        st.write(f"🔑 키워드: {res['keywords']}")
    
    st.json(res['detail'])
    
    if st.button("💾 DB 저장"):
        save_db(res['channel'], res['title'], res['prob'], res['url'], res['keywords'], res['detail'])

    st.write("---")
    st.write("### 🔍 유사 영상 추천")
    similar_videos = get_similar_content(res['keywords'])
    
    if similar_videos:
        for vid in similar_videos:
            st.info(f"📄 **{vid['title']}** (유사도: {int(vid['score']*100)}%)")
    else:
        st.caption("유사한 영상 없음")

# -----------------------------------------------------------------------------
# 4. 관리자 메뉴
# -----------------------------------------------------------------------------
st.divider()
with st.expander("🔐 관리자 기능"):
    if st.session_state["is_admin"]:
        
        # A. 데이터 복구
        uploaded_file = st.file_uploader("백업 CSV 업로드", type="csv")
        if uploaded_file and st.button("🚨 데이터 복구 시작"):
            try:
                df = pd.read_csv(uploaded_file)
                bar = st.progress(0)
                success_count = 0
                
                for i, row in df.iterrows():
                    title = str(row.get('video_title', ''))
                    if not title or title == 'nan': continue
                    
                    data = {
                        "analysis_date": str(row.get('analysis_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))),
                        "channel_name": str(row.get('channel_name', 'Unknown')),
                        "video_title": title,
                        "fake_prob": int(row['fake_prob']) if pd.notna(row.get('fake_prob')) else 0,
                        "video_url": str(row.get('video_url', '')),
                        "keywords": str(row.get('keywords', '')),
                        "detail_json": {"summary": "복구됨"},
                        "vector_json": None 
                    }
                    try:
                        supabase.table("analysis_history").insert(data).execute()
                        success_count += 1
                    except: pass
                    bar.progress(int(((i+1)/len(df))*100))
                
                st.success(f"✅ {success_count}건 복구 완료!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"복구 에러: {e}")

        st.write("---")

        # B. 강제 업데이트 (Gemini 사용)
        if st.button("♻️ AI 학습 강제 실행 (Gemini)"):
            progress_text = st.empty()
            bar = st.progress(0)
            
            try:
                try:
                    target_rows = supabase.table("analysis_history").select("*").is_("vector_json", "null").execute().data
                except:
                    target_rows = supabase.table("analysis_history").select("*").execute().data

                total = len(target_rows)
                st.write(f"🎯 학습 대상: {total}건")
                
                for i, row in enumerate(target_rows):
                    txt = f"{row.get('keywords','')} {row.get('video_title','')}"
                    try:
                        vec = vector_engine.get_embedding(txt)
                        if vec:
                            supabase.table("analysis_history").update({"vector_json": vec}).eq("id", row['id']).execute()
                    except: pass
                    
                    bar.progress(int(((i+1)/total)*100))
                    progress_text.text(f"학습 중... {i+1}/{total}")
                
                st.success("✅ 학습 완료!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"업데이트 에러: {e}")
