import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from supabase import create_client
import openai

# -----------------------------------------------------------------------------
# 1. 설정 및 초기화 (Setup)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI 영상 분석기", layout="wide", page_icon="🎬")

# 비밀번호 및 API 키 로드 (st.secrets 사용)
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except:
    st.error("❌ .streamlit/secrets.toml 파일에 API 키가 설정되지 않았습니다.")
    st.stop()

# 클라이언트 연결
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 세션 상태 초기화
if "is_admin" not in st.session_state: st.session_state["is_admin"] = False
if "analysis_result" not in st.session_state: st.session_state["analysis_result"] = None

# -----------------------------------------------------------------------------
# 2. 유틸리티 클래스 & 함수 (Utils)
# -----------------------------------------------------------------------------

class VectorEngine:
    """텍스트를 벡터로 변환하는 엔진"""
    def get_embedding(self, text):
        try:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"벡터 생성 실패: {e}")
            return None

vector_engine = VectorEngine()

def get_similar_content(current_keywords):
    """
    [핵심 수정] analysis_history 테이블에서 유사한 과거 영상을 찾습니다.
    """
    try:
        if not current_keywords: return []
        
        # 1. 현재 분석 키워드의 벡터 생성
        query_vector = vector_engine.get_embedding(current_keywords)
        if not query_vector: return []

        # 2. DB에서 벡터 데이터 가져오기 (Target: analysis_history)
        response = supabase.table("analysis_history").select("video_title, video_url, vector_json").not_.is_("vector_json", "null").execute()
        
        candidates = []
        for row in response.data:
            # 벡터 파싱
            if isinstance(row['vector_json'], str):
                vec = json.loads(row['vector_json'])
            else:
                vec = row['vector_json']
            
            if not vec: continue

            # 코사인 유사도 계산
            dot_product = np.dot(query_vector, vec)
            norm_a = np.linalg.norm(query_vector)
            norm_b = np.linalg.norm(vec)
            similarity = dot_product / (norm_a * norm_b)

            if similarity > 0.6: # 유사도 60% 이상만 추천
                candidates.append({
                    "title": row['video_title'],
                    "url": row['video_url'],
                    "score": similarity
                })
        
        # 점수 높은 순 정렬 후 상위 3개 반환
        return sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]
        
    except Exception as e:
        print(f"유사도 검색 에러: {e}")
        return []

def save_db(ch, ti, pr, url, kw, detail):
    """
    [핵심 수정] analysis_history 테이블에 데이터를 저장합니다.
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
            "vector_json": embedding # 벡터도 같이 저장
        }
        
        # 저장 (Target: analysis_history)
        supabase.table("analysis_history").insert(data_to_insert).execute()
        st.toast("✅ DB 저장 및 학습 완료!", icon="💾")
        time.sleep(1) # 사용자 확인용 대기
        
    except Exception as e: 
        st.error(f"❌ 데이터베이스 저장 실패: {e}")

# -----------------------------------------------------------------------------
# 3. 메인 UI (Main Interface)
# -----------------------------------------------------------------------------
st.title("🎬 AI 유튜브 분석기 (Unified v3)")

# 사이드바 (로그인)
with st.sidebar:
    st.header("설정")
    if not st.session_state["is_admin"]:
        pwd = st.text_input("관리자 비밀번호", type="password")
        if st.button("로그인"):
            if pwd == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True
                st.rerun()
            else:
                st.error("비밀번호가 틀렸습니다.")
    else:
        st.success("관리자 로그인 중")
        if st.button("로그아웃"):
            st.session_state["is_admin"] = False
            st.rerun()

# 메인 기능: URL 입력 및 분석
url_input = st.text_input("유튜브 URL을 입력하세요", placeholder="https://youtu.be/...")

if st.button("🚀 분석 시작", type="primary"):
    if not url_input:
        st.warning("URL을 입력해주세요.")
    else:
        # [중복 체크] analysis_history 테이블 확인
        try:
            check = supabase.table("analysis_history").select("*").eq("video_url", url_input).execute()
            if check.data:
                st.info("💡 이미 분석된 영상입니다. (DB 데이터 로드)")
                res = check.data[0]
                # DB 데이터를 세션에 저장 (화면 표시용)
                st.session_state["analysis_result"] = {
                    "channel": res['channel_name'],
                    "title": res['video_title'],
                    "prob": res['fake_prob'],
                    "keywords": res['keywords'],
                    "detail": res['detail_json'],
                    "url": res['video_url']
                }
            else:
                # [신규 분석] (실제 AI 로직 대신 더미 데이터 사용 예시)
                with st.spinner("AI가 영상을 분석 중입니다..."):
                    time.sleep(1.5) # 분석 흉내
                    # --- 실제로는 여기서 LLM/YouTube API 호출 ---
                    result_data = {
                        "channel": "테스트 채널",
                        "title": "테스트 영상 제목 (분석됨)",
                        "prob": 88, # 가짜 확률
                        "keywords": "AI, 테스트, 데이터복구",
                        "detail": {"summary": "이 영상은 테스트용입니다."},
                        "url": url_input
                    }
                    st.session_state["analysis_result"] = result_data
                    
                    # 자동 저장을 원하면 여기서 save_db 호출 (선택사항)
                    
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# 결과 화면 표시
if st.session_state["analysis_result"]:
    res = st.session_state["analysis_result"]
    
    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("🚫 조작 의심 확률", f"{res['prob']}%")
        if res['prob'] > 70: st.error("⚠️ 주의 필요")
        else: st.success("✅ 양호함")
    
    with c2:
        st.subheader(res['title'])
        st.write(f"📺 채널: **{res['channel']}**")
        st.write(f"🔑 키워드: {res['keywords']}")
    
    st.json(res['detail'])
    
    # 저장 버튼
    if st.button("💾 데이터베이스에 저장"):
        save_db(res['channel'], res['title'], res['prob'], res['url'], res['keywords'], res['detail'])

    # [벡터 검색] 유사 영상 추천
    st.write("---")
    st.write("### 🔍 유사한 과거 분석 사례")
    similar_videos = get_similar_content(res['keywords'])
    
    if similar_videos:
        for vid in similar_videos:
            st.info(f"📄 **{vid['title']}** (유사도: {int(vid['score']*100)}%)")
    else:
        st.caption("유사한 영상이 없습니다.")

# -----------------------------------------------------------------------------
# 4. 관리자 메뉴 (Admin - 복구 및 업데이트)
# -----------------------------------------------------------------------------
st.divider()
with st.expander("🔐 관리자 (시스템 복구 및 관리)"):
    if st.session_state["is_admin"]:
        st.write("### 🚑 데이터 복구 & AI 학습 센터")
        
        # A. 데이터 복구 (CSV -> DB)
        uploaded_file = st.file_uploader("백업 CSV 파일 업로드", type="csv")
        if uploaded_file and st.button("🚨 데이터 복구 시작 (analysis_history)"):
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
                        "vector_json": None # 일단 비워둠 (업데이트에서 채움)
                    }
                    try:
                        supabase.table("analysis_history").insert(data).execute()
                        success_count += 1
                    except: pass
                    bar.progress(int(((i+1)/len(df))*100))
                
                st.success(f"✅ {success_count}건 복구 완료! 아래 업데이트 버튼을 누르세요.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"복구 에러: {e}")

        st.write("---")

        # B. 강제 업데이트 (AI 학습)
        if st.button("♻️ AI 학습 강제 실행 (벡터 생성)"):
            progress_text = st.empty()
            bar = st.progress(0)
            
            try:
                # 학습 안 된 데이터 조회 (오류 시 전체 조회)
                try:
                    target_rows = supabase.table("analysis_history").select("*").is_("vector_json", "null").execute().data
                except:
                    target_rows = supabase.table("analysis_history").select("*").execute().data

                total = len(target_rows)
                if total == 0:
                    st.info("모든 데이터가 이미 학습되었습니다.")
                else:
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
