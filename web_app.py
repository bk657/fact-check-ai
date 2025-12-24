import streamlit as st
import google.generativeai as genai
import time
import pandas as pd
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 시스템 설정 ---
st.set_page_config(page_title="Key A 전수 조사 (All-Model Test)", page_icon="🧬", layout="wide")

st.title("🧬 Key A : 모든 모델 생존 및 응답 테스트")
st.markdown("""
이 도구는 **Key A**를 사용하여 구글의 모든 Gemini 모델에게 실제 데이터를 전송합니다.
어떤 모델이 살아있고, 어떤 모델이 '사용량 초과(429)'인지 한눈에 파악할 수 있습니다.
""")

# 1. Key A 로드
try:
    API_KEY = st.secrets["GOOGLE_API_KEY_A"]
    genai.configure(api_key=API_KEY)
    st.success(f"🔑 Key A 로드 완료 (사용 준비됨)")
except Exception as e:
    st.error("❌ secrets.toml 파일에서 GOOGLE_API_KEY_A를 찾을 수 없습니다.")
    st.stop()

# 2. 테스트용 더미 데이터 (실제 상황 시뮬레이션)
TEST_TITLE = "나비약(디에타민) 부작용과 위험성, 절대 먹지 마세요"
TEST_TRANSCRIPT = """
여러분 안녕하세요. 오늘은 다이어트 약으로 알려진 나비약, 즉 디에타민에 대해 이야기해보려 합니다.
이 약은 식욕 억제제로 쓰이지만 사실 마약류로 분류되는 향정신성 의약품입니다.
부작용으로는 심장 두근거림, 불면증, 그리고 심각할 경우 환청과 망상까지 겪을 수 있습니다.
최근 경찰 조사 결과에 따르면 이 약을 불법으로 처방받아 되파는 사례도 늘고 있다고 하는데요...
(이하 생략 - 테스트를 위해 300자 정도만 보냅니다)
"""

# 3. 안전 설정 해제 (필터링으로 인한 실패 방지)
safety_settings_none = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

if st.button("🚀 전수 조사 시작 (Start Diagnosis)", use_container_width=True):
    st.divider()
    
    # [1단계] 사용 가능한 모델 리스트 가져오기
    st.write("🔍 **1단계: 접근 가능한 모델 목록 조회 중...**")
    candidate_models = []
    try:
        for m in genai.list_models():
            # 텍스트 생성이 가능한 'gemini' 모델만 필터링
            if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
                candidate_models.append(m.name)
    except Exception as e:
        st.error(f"모델 목록 조회 실패: {e}")
        st.stop()
        
    st.info(f"총 {len(candidate_models)}개의 Gemini 모델을 발견했습니다. 순차적으로 테스트를 진행합니다.")
    
    # [2단계] 모델별 실제 요청 테스트
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, model_name in enumerate(candidate_models):
        status_text.text(f"Testing... {model_name}")
        progress_bar.progress((idx + 1) / len(candidate_models))
        
        start_time = time.time()
        result_status = "❌ 실패"
        detail = ""
        
        try:
            model = genai.GenerativeModel(model_name)
            
            # 실제 프롬프트 전송
            prompt = f"""
            [Test Request]
            Video Title: {TEST_TITLE}
            Transcript: {TEST_TRANSCRIPT}
            Task: Extract one keyword.
            """
            
            response = model.generate_content(prompt, safety_settings=safety_settings_none)
            
            if response.text:
                result_status = "✅ 성공"
                detail = response.text.strip()[:20] + "..." # 결과 일부만 표시
            else:
                result_status = "⚠️ 빈 응답"
                detail = "응답 텍스트 없음"
                
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "Quota" in err_msg:
                result_status = "⛔ 429 (한도 초과)"
                detail = "일일/분당 사용량 초과"
            elif "404" in err_msg:
                result_status = "🗑️ 404 (찾을 수 없음)"
                detail = "모델이 존재하지 않거나 폐기됨"
            else:
                result_status = "❌ 에러"
                detail = err_msg[:50]
        
        elapsed = round(time.time() - start_time, 2)
        results.append({
            "모델명": model_name,
            "상태": result_status,
            "소요시간": f"{elapsed}초",
            "상세 내용": detail
        })
        
        # API 부하 방지를 위해 약간 대기
        time.sleep(1)

    # [3단계] 결과 리포트 출력
    st.divider()
    st.subheader("📊 진단 최종 결과")
    
    df = pd.DataFrame(results)
    
    # 스타일링: 성공은 초록색, 429는 빨간색
    def highlight_status(val):
        if "✅" in val: return 'background-color: #d4edda; color: #155724' # Green
        elif "⛔" in val: return 'background-color: #f8d7da; color: #721c24' # Red
        return ''

    st.dataframe(df.style.applymap(highlight_status, subset=['상태']), use_container_width=True, height=600)
    
    # 추천 모델 찾기
    success_models = [r['모델명'] for r in results if "✅" in r['상태']]
    
    if success_models:
        st.success("🎉 **사용 가능한 모델이 발견되었습니다!** 아래 모델 중 하나를 코드에 적용하세요.")
        st.code("\n".join(success_models))
        
        # 
    else:
        st.error("😱 **모든 모델이 실패했습니다.** 현재 Key A는 완전히 쿼터가 차단되었거나 문제가 있습니다.")
