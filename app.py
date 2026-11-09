import os
import streamlit as st
import streamlit.components.v1 as components
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# -------------------------------------------------------------------
# 1. 모바일 최적화 페이지 설정 및 CSS
# -------------------------------------------------------------------
st.set_page_config(page_title="복음방 스피치 훈련", page_icon="📖", layout="centered")

st.markdown("""
    <style>
    /* 스마트폰 화면에 맞춘 텍스트 크기 및 여백 조정 */
    .title-text { font-size: 1.5rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-text { font-size: 0.9rem; color: #4B5563; margin-bottom: 1rem; }
    .result-box { background-color: #F9FAFB; padding: 1rem; border-radius: 0.5rem; border: 1px solid #E5E7EB; line-height: 1.6; font-size: 0.95rem; }
    .stTextArea textarea { font-size: 1rem !important; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. 브라우저 마이크 권한 감지 컴포넌트
# -------------------------------------------------------------------
def inject_mic_permission_checker():
    js_code = """
    <script>
    async function requestMicPermission() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
            window.parent.postMessage({type: 'MIC_STATUS', status: 'granted'}, '*');
        } catch (err) {
            window.parent.postMessage({type: 'MIC_STATUS', status: 'denied'}, '*');
        }
    }
    requestMicPermission();
    </script>
    """
    components.html(js_code, height=0, width=0)

if "mic_granted" not in st.session_state:
    st.session_state["mic_granted"] = None

inject_mic_permission_checker()

# -------------------------------------------------------------------
# 3. 데이터 및 Pydantic 스키마 정의
# -------------------------------------------------------------------
# 1강 원본 데이터 (절대 변경되지 않는 기준)
LECTURE_1_TEXT = """
[도입] 
전 세계 약 81억 인구 중 약 81%, 66억 명이 종교를 가지고 있습니다. 많은 사람이 종교를 가지는 이유를 알기 위해 종교의 의미와 경서에 대해 알아보겠습니다.
[본론] 
1. 종교의 의미: '으뜸 종'과 '가르칠 교'로 이루어져 '하늘의 것을 가르치는 교육' 즉 '으뜸가는 가르침'입니다.
2. 종교의 역할: 'Religion(릴리전)'은 'Re(다시) + Ligare(연결하다)'로 다시 연결하다는 뜻입니다. 죄 때문에 끊어진 하나님과 사람 사이의 생명줄을 다시 이어주는 것이 종교의 역할입니다. 이 죄를 해결하기 위해 예수님이 십자가에서 피 흘려 돌아가셨고, 이 피로 죄 사함을 받아 생명이신 하나님과 다시 연결되어 구원을 얻게 됩니다.
3. 참 종교의 경서: 기독교 성경이 참 종교의 경서인 이유는 예언과 그 예언이 그대로 이루어진 실체가 있기 때문입니다. 성경을 통해 살아계신 하나님의 뜻과 진리를 아는 것이 신앙인이 가장 먼저 해야 할 일입니다(호 6:6, 딤전 2:4).
[결론] 
종교는 으뜸가는 가르침이며, 그 역할은 죄 사함을 통해 떠나가신 하나님과 사람을 다시 이어 생명을 얻게 하는 것입니다. 예언과 실체가 있는 기독교 성경을 통해 하나님의 약속하신 복을 받기를 바랍니다.
"""

KEYWORDS = ["으뜸가는 가르침", "생명줄", "예수님의 피", "죄 사함", "예언과 실체"]

class KeywordCheck(BaseModel):
    keyword: str = Field(description="검증 대상 핵심 단어")
    is_included: bool = Field(description="발표 내용에 포함되었는지 여부 (True/False)")

class SpeechEvaluation(BaseModel):
    score: int = Field(description="스피치 정확도 점수 (0~100점)")
    flow_feedback: str = Field(description="도입-본론-결론의 흐름을 잘 따라갔는지에 대한 피드백 (외부 지식 차단, 오직 원문 기준)")
    keyword_checks: list[KeywordCheck] = Field(description="핵심 단어 포함 여부 체크리스트")
    highlighted_result: str = Field(
        description="교정 결과 텍스트 (빠지거나 틀린 단어는 <span style='color:red; font-weight:bold;'>단어</span>, 원본에 없는데 불필요하게 추가된 말은 <span style='color:blue; font-weight:bold;'>단어</span>로 하이라이팅)"
    )

# -------------------------------------------------------------------
# 4. Gemini API 엄격 평가 로직
# -------------------------------------------------------------------
def evaluate_speech(api_key: str, user_text: str) -> SpeechEvaluation:
    client = genai.Client(api_key=api_key)
    
    system_instruction = """
    당신은 복음방 스피치 훈련을 돕는 엄격하고 정확한 AI 튜터입니다.

    [절대 규칙]
    1. 사용자의 스피치 텍스트를 오직 제공된 [기준 원본 교안]과만 비교하세요. 외부의 신학적 지식, 성경 구절, 배경지식은 1%도 개입시키지 마세요.
    2. 도입 -> 본론 -> 결론의 흐름을 잘 지켰는지 평가하세요.
    3. 빠지거나 틀린 핵심 내용은 <span style="color:red; font-weight:bold;">단어</span>로, 원본에 없는데 사용자가 지어내거나 추가한 외부 내용은 <span style="color:blue; font-weight:bold;">단어</span>로 정확히 하이라이팅하세요.
    """
    
    prompt = f"[기준 원본 교안 (1강)]:\n{LECTURE_1_TEXT}\n\n[필수 키워드]:\n{', '.join(KEYWORDS)}\n\n[사용자 발표 텍스트]:\n{user_text}\n\n위 기준에 따라 사용자의 발표를 엄격하게 평가해 주세요."
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=SpeechEvaluation,
            temperature=0.0, # 창의성 0으로 고정 (엄격한 원본 대조)
        ),
    )
    return SpeechEvaluation.model_validate_json(response.text)

# -------------------------------------------------------------------
# 5. 메인 UI 및 마이크 권한 차단
# -------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("Gemini API Key:", type="password")
    
    st.divider()
    st.caption("권한 상태 (테스트용)")
    if st.button("✅ 권한 허용 상태로 변경"): st.session_state["mic_granted"] = True
    if st.button("❌ 권한 거부 상태로 변경"): st.session_state["mic_granted"] = False

# 마이크 권한 거부 시
if st.session_state["mic_granted"] is False:
    st.error("🚫 마이크 권한이 거부되었습니다.")
    st.warning("스피치 진단을 위해 브라우저 설정에서 마이크를 '허용'해 주세요.")
    if st.button("🔄 다시 시도", type="primary"): st.rerun()
    st.stop()

# 메인 화면
st.markdown('<div class="title-text">📖 복음방 스피치 훈련기</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">스마트폰 마이크를 켜고 1강 요약 발표를 진행해 보세요.</div>', unsafe_allow_html=True)

with st.expander("📝 1강 요약 기준 보기 (종교란 무엇인가)"):
    st.write(LECTURE_1_TEXT)
    st.caption(f"**필수 키워드:** {', '.join(KEYWORDS)}")

# 사용자 입력 영역 (스마트폰 음성 입력 대응)
st.subheader("🎙️ 요약 스피치")
user_speech_input = st.text_area(
    "마이크(받아쓰기) 버튼을 눌러 발표하거나 텍스트를 입력하세요.", 
    placeholder="예: 종교는 으뜸가는 가르침입니다...",
    height=150
)

if st.button("🚀 스피치 진단하기", type="primary", use_container_width=True):
    if not api_key:
        st.error("❌ 사이드바 메뉴(왼쪽 상단 ❯ 기호)에서 API Key를 입력해 주세요.")
    elif not user_speech_input.strip():
        st.warning("⚠️ 발표 내용을 입력해 주세요.")
    else:
        with st.spinner("AI가 교안과 흐름을 대조하며 분석 중입니다..."):
            try:
                res = evaluate_speech(api_key, user_speech_input)
                
                st.divider()
                st.subheader("📊 진단 결과")
                st.metric("종합 정확도", f"{res.score}점")
                
                st.markdown("##### 🌊 흐름 피드백")
                st.info(res.flow_feedback)
                
                st.markdown("##### 🔑 키워드 포함 여부")
                cols = st.columns(3)
                for idx, item in enumerate(res.keyword_checks):
                    status = "✅" if item.is_included else "❌"
                    cols[idx % 3].write(f"{status} {item.keyword}")
                
                st.markdown("##### 📝 실시간 텍스트 교정")
                st.caption("🔴 **빨간색**: 빠지거나 틀린 내용 | 🔵 **파란색**: 원본에 없는 추가 내용(외부 지식)")
                st.markdown(f'<div class="result-box">{res.highlighted_result}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")