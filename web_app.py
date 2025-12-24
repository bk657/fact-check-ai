import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

st.set_page_config(page_title="Key A 진단실", page_icon="🏥")

st.title("🏥 Key A 정밀 진단 리포트")

# 1. Key A 로드 확인
try:
    API_KEY = st.secrets["GOOGLE_API_KEY_A"]
    st.success(f"🔑 Key A 로드 성공 (키 길이: {len(API_KEY)})")
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"❌ Key A 로드 실패: {e}")
    st.stop()

if st.button("🚀 진단 시작"):
    st.divider()
    
    # [진단 1] 이 키로 접근 가능한 모델 목록 조회
    st.subheader("1️⃣ 사용 가능한 모델 목록 (List Models)")
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        if available_models:
            st.write("📋 **Google이 허용한 모델 리스트:**")
            st.code("\n".join(available_models))
        else:
            st.error("❌ 접근 가능한 모델이 하나도 없습니다. (프로젝트 설정 문제)")
    except Exception as e:
        st.error(f"❌ 모델 목록 조회 실패: {e}")
        st.stop()

    # [진단 2] 주요 모델별 'Hello' 통신 테스트
    st.divider()
    st.subheader("2️⃣ 주요 모델 생존 테스트 (Ping Test)")
    
    targets = [
        "models/gemini-2.0-flash", 
        "models/gemini-1.5-flash", 
        "models/gemini-1.5-pro",
        "models/gemini-pro"
    ]
    
    for model_name in targets:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**{model_name}**")
        with col2:
            if model_name not in available_models:
                st.warning("⚠️ 목록에 없음 (사용 불가)")
                continue
                
            try:
                model = genai.GenerativeModel(model_name)
                # 안전 장치 해제하고 아주 짧은 인사만 보냄
                response = model.generate_content("hi", safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                })
                if response.text:
                    st.success(f"✅ 정상 작동 (응답: {response.text.strip()})")
                else:
                    st.warning("⚠️ 응답 없음 (빈 텍스트)")
            except Exception as e:
                st.error(f"❌ 에러 발생: {e}")

    st.info("💡 위 결과에서 '✅ 정상 작동'이 뜬 모델 이름을 알려주세요. 그것만 써야 합니다.")
