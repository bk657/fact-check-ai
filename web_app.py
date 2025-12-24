import streamlit as st
import google.generativeai as genai
import time

st.set_page_config(page_title="API Key Diagnostic", page_icon="🩺")
st.title("🩺 Google Gemini API 정밀 진단")

# 1. 키 로드 확인
try:
    KEY_A = st.secrets["GOOGLE_API_KEY_A"]
    KEY_B = st.secrets["GOOGLE_API_KEY_B"]
    st.success("✅ secrets.toml에서 키 2개를 찾았습니다.")
except:
    st.error("❌ secrets.toml 파일에 키가 없습니다.")
    st.stop()

# 2. Key B 테스트 함수
def test_key(api_key, label):
    st.divider()
    st.subheader(f"🔑 {label} 테스트 시작")
    genai.configure(api_key=api_key)
    
    # [테스트 1] 모델 리스트 조회 (연결 확인)
    st.write("1️⃣ 모델 리스트 조회 중...")
    try:
        models = [m.name for m in genai.list_models()]
        st.success(f"✅ 연결 성공! 사용 가능한 모델: {len(models)}개")
        with st.expander("모델 목록 보기"):
            st.write(models)
            
        # 1.5-flash가 있는지 확인
        if 'models/gemini-1.5-flash' in models:
            st.info("👌 'gemini-1.5-flash' 모델이 목록에 있습니다.")
        else:
            st.error("😱 'gemini-1.5-flash' 모델이 목록에 없습니다! (이게 원인입니다)")
            return # 모델이 없으면 중단
            
    except Exception as e:
        st.error(f"❌ 연결 실패 (Auth 문제): {e}")
        return

    # [테스트 2] 단순 문자열 추론 (기능 확인)
    st.write("2️⃣ 단순 '안녕' 테스트 중...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content("안녕")
        st.success(f"✅ 응답 성공: {res.text}")
    except Exception as e:
        st.error(f"❌ 단순 추론 실패: {e}")
        return

    # [테스트 3] 대용량 데이터 전송 (데이터 양 문제 확인)
    st.write("3️⃣ 대용량(3만 자) 데이터 전송 테스트...")
    try:
        # 의미 없는 3만 자 텍스트 생성
        dummy_text = "테스트 데이터 " * 5000 
        prompt = f"이 텍스트의 길이를 요약해줘: {dummy_text[:30000]}"
        
        res = model.generate_content(prompt)
        st.success(f"✅ 대용량 처리 성공: {res.text}")
        st.balloons()
    except Exception as e:
        if "400" in str(e):
            st.error("❌ 데이터 양 과부하 (400 Bad Request)")
        elif "429" in str(e):
            st.error("❌ 속도 제한 초과 (429 Rate Limit)")
        else:
            st.error(f"❌ 기타 에러 발생: {e}")

# 실행 버튼
col1, col2 = st.columns(2)
with col1:
    if st.button("Key A (수사관) 테스트"):
        test_key(KEY_A, "Key A")
with col2:
    if st.button("Key B (판사) 테스트"):
        test_key(KEY_B, "Key B")
