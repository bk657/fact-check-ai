import streamlit as st
from supabase import create_client
import time

st.title("🔌 Supabase 연결 정밀 진단")

# 1. 시크릿 로드 확인
st.write("### 1. 설정(Secrets) 확인")
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    st.success(f"✅ URL 로드됨: {SUPABASE_URL[:15]}...")
    st.success(f"✅ KEY 로드됨: {SUPABASE_KEY[:10]}...")
except Exception as e:
    st.error(f"❌ 시크릿 로드 실패: {e}")
    st.stop()

# 2. 클라이언트 생성
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"❌ 클라이언트 생성 실패: {e}")
    st.stop()

st.divider()

# 3. 쓰기 테스트
st.write("### 2. 데이터 쓰기 테스트")
if st.button("🚀 데이터 전송 시도"):
    try:
        test_msg = f"Test Message at {time.strftime('%H:%M:%S')}"
        
        # debug_test 테이블에 insert 시도
        data = {"message": test_msg}
        response = supabase.table("debug_test").insert(data).execute()
        
        st.success("✅ 성공! 데이터가 저장되었습니다.")
        st.write("응답 결과:", response.data)
        
    except Exception as e:
        st.error("❌ 저장 실패 (이게 뜨면 DB 연결 문제임)")
        st.code(str(e))

# 4. 읽기 테스트
st.write("### 3. 데이터 읽기 테스트")
if st.button("📂 데이터 조회 시도"):
    try:
        response = supabase.table("debug_test").select("*").order("id", desc=True).limit(5).execute()
        st.write(response.data)
        if response.data:
            st.success("✅ 읽기 성공!")
        else:
            st.warning("데이터가 비어있습니다 (쓰기 먼저 하세요)")
    except Exception as e:
        st.error(f"❌ 읽기 실패: {e}")
