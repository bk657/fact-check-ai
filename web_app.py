import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time

# 페이지 설정
st.set_page_config(page_title="Key A Diagnostic Tool", page_icon="🩺", layout="centered")

st.title("🩺 Key A 정밀 진단 모드")
st.write("이 도구는 Key A의 연결, 권한, 안전 필터, 데이터 한계를 테스트합니다.")

# 1. 시크릿 로드 확인
try:
    API_KEY_A = st.secrets["GOOGLE_API_KEY_A"]
    st.success("✅ secrets.toml에서 'GOOGLE_API_KEY_A'를 찾았습니다.")
except:
    st.error("❌ 'GOOGLE_API_KEY_A'가 설정되지 않았습니다.")
    st.stop()

# 진단 시작 버튼
if st.button("🚀 진단 시작 (Key A)"):
    
    # --- [테스트 1] 모델 리스트 조회 (연결/권한 확인) ---
    st.divider()
    st.subheader("1️⃣ 연결 및 모델 권한 테스트")
    genai.configure(api_key=API_KEY_A)
    
    available_models = []
    try:
        with st.spinner("구글 서버와 통신 중..."):
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        
        st.success(f"✅ 연결 성공! (접근 가능한 모델: {len(available_models)}개)")
        
        # 주요 모델 존재 여부 체크
        target_models = ['models/gemini-1.5-flash', 'models/gemini-2.0-flash', 'models/gemini-pro']
        for tm in target_models:
            if tm in available_models:
                st.info(f"👌 {tm}: 사용 가능")
            else:
                st.warning(f"⚠️ {tm}: 목록에 없음 (사용 불가)")
                
    except Exception as e:
        st.error(f"❌ [치명적 오류] 연결 실패: {e}")
        st.stop() # 연결 안 되면 뒤에는 의미 없음

    # --- [테스트 2] 안전 필터(Safety Filter) 테스트 ---
    st.divider()
    st.subheader("2️⃣ 마약/범죄 키워드 필터링 테스트")
    
    # 안전 장치 해제 설정
    safety_settings_none = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    test_prompt = "나비약(디에타민, 펜터민)의 부작용과 위험성에 대해 설명해줘. 마약류 관리법 위반 사례도 포함해."
    st.caption(f"테스트 프롬프트: {test_prompt}")
    
    # 사용 가능한 첫 번째 모델로 테스트
    test_model_name = 'gemini-1.5-flash' if 'models/gemini-1.5-flash' in available_models else available_models[0].replace('models/', '')
    st.write(f"🧪 테스트 모델: **{test_model_name}**")
    
    try:
        model = genai.GenerativeModel(test_model_name)
        response = model.generate_content(test_prompt, safety_settings=safety_settings_none)
        
        if response.text:
            st.success("✅ 안전 필터 통과! (답변 생성됨)")
            with st.expander("답변 내용 보기"):
                st.write(response.text)
        else:
            st.error("❌ 답변 생성 실패 (빈 응답). 안전 필터에 걸렸을 수 있습니다.")
            st.write(response.prompt_feedback) # 차단 원인 출력
            
    except Exception as e:
        st.error(f"❌ 에러 발생: {e}")

    # --- [테스트 3] 대용량 데이터(자막) 처리 테스트 ---
    st.divider()
    st.subheader("3️⃣ 대용량 컨텍스트 처리 테스트")
    
    # 3만 자 더미 데이터 생성
    dummy_transcript = "나비약 펜터민 부작용 " * 3000  # 약 3~4만 자
    st.write(f"📦 데이터 크기: {len(dummy_transcript)}자 전송 시도...")
    
    large_prompt = f"""
    이 긴 텍스트에서 핵심 검색어를 1개 추출해줘.
    [Text]: {dummy_transcript}
    """
    
    try:
        start_time = time.time()
        response = model.generate_content(large_prompt, safety_settings=safety_settings_none)
        end_time = time.time()
        
        st.success(f"✅ 대용량 처리 성공! (소요 시간: {end_time - start_time:.2f}초)")
        st.write(f"응답: {response.text}")
        
    except Exception as e:
        st.error(f"❌ 대용량 처리 실패: {e}")
        if "429" in str(e):
            st.warning("👉 원인: 사용량 초과 (Rate Limit Exceeded)")
        elif "400" in str(e):
            st.warning("👉 원인: 잘못된 요청 (토큰 한도 초과 등)")

st.write("---")
st.info("💡 이 결과를 복사해서 알려주시면 바로 해결책을 드릴 수 있습니다.")
