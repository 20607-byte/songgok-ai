import os

# Streamlit Cloud(Linux 서버)에서 Orange3 임포트 시 GUI/디스플레이 오류 방지
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pickle
from pathlib import Path
import numpy as np
import streamlit as st
import Orange


# ---------------------------------------------------------
# 1. 페이지 기본 설정 및 스타일 정의
# ---------------------------------------------------------
st.set_page_config(
    page_title="어린이 보호구역 교통안전 예측 신호등",
    page_icon="🚦",
    layout="centered",
)

st.markdown(
    """
    <style>
    .traffic-box {
        background-color: #1e222a;
        border-radius: 24px;
        padding: 28px 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    .lights-row {
        display: flex;
        justify-content: center;
        gap: 24px;
        margin: 14px 0 18px 0;
    }
    .bulb {
        width: 78px;
        height: 78px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        background-color: #2f3542;
        opacity: 0.22;
        transition: all 0.25s ease;
    }
    .bulb.active-green {
        background-color: #2ed573;
        opacity: 1;
        box-shadow: 0 0 28px #2ed573;
    }
    .bulb.active-yellow {
        background-color: #ffa502;
        opacity: 1;
        box-shadow: 0 0 28px #ffa502;
    }
    .bulb.active-red {
        background-color: #ff4757;
        opacity: 1;
        box-shadow: 0 0 28px #ff4757;
    }
    .status-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .status-sub {
        color: #ced6e0;
        font-size: 15px;
    }
    .safety-card {
        border-radius: 16px;
        padding: 22px;
        border-left: 8px solid;
        background-color: #f8f9fa;
        color: #1e272e;
        margin-top: 12px;
    }
    .safety-card h4 {
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 19px;
    }
    .safety-card ul {
        margin-bottom: 0;
        padding-left: 20px;
        line-height: 1.75;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. 오렌지3 모델(.pkcls) 로드
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "traffic_casualty_model.pkcls"


@st.cache_resource
def load_orange_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


if not MODEL_PATH.exists():
    st.error(
        f"모델 파일을 찾을 수 없습니다: `{MODEL_PATH.name}`\n\n"
        "GitHub 저장소에 `app.py`와 같은 경로에 `traffic_casualty_model.pkcls` 파일을 업로드해주세요."
    )
    st.stop()

try:
    model = load_orange_model(MODEL_PATH)
except Exception as e:
    st.error(f"모델 로드 중 오류가 발생했습니다: {e}")
    st.stop()


# ---------------------------------------------------------
# 3. 실시간 예측 함수
# ---------------------------------------------------------
def predict_casualties(
    orange_model,
    sido: str,
    accident_count: int,
    minor_injury_count: int,
    severe_injury_count: float | None = None,
    reported_injury_count: float | None = None,
) -> float:
    """
    오렌지3 모델의 domain.attributes 순서에 맞춰 입력 Table을 생성하고
    예상 사망자 수를 반환합니다.
    """
    row = []
    for attr in orange_model.domain.attributes:
        if attr.name == "시도":
            row.append(sido)
        elif attr.name == "사고건수":
            row.append(float(accident_count))
        elif attr.name == "경상자수":
            row.append(float(minor_injury_count))
        elif attr.name == "중상자수":
            row.append(float(severe_injury_count) if severe_injury_count is not None else np.nan)
        elif attr.name == "부상신고자수":
            row.append(float(reported_injury_count) if reported_injury_count is not None else np.nan)
        else:
            row.append(np.nan)

    input_table = Orange.data.Table.from_list(orange_model.domain, [row])
    prediction = orange_model(input_table)
    return max(0.0, float(prediction[0]))


# ---------------------------------------------------------
# 4. 입력 UI (시도 선택 및 슬라이더)
# ---------------------------------------------------------
st.title("🚦 어린이 보호구역 교통안전 예측 신호등")
st.caption("오렌지3(Orange3) 학습 모델을 기반으로 지역별 교통사고 발생 규모에 따른 예상 사망자 수와 안전 수칙을 안내합니다.")

# 모델에 학습된 시도 목록 자동 추출 (기본값: 서울, 경기, 부산)
if "시도" in model.domain:
    sido_options = list(model.domain["시도"].values)
else:
    sido_options = ["서울", "경기", "부산"]

default_sido_idx = sido_options.index("서울") if "서울" in sido_options else 0

with st.container(border=True):
    st.subheader("1️⃣ 지역 및 사고 지표 입력")

    selected_sido = st.selectbox(
        "시도 선택",
        options=sido_options,
        index=default_sido_idx,
        help="모델이 학습한 시도 단위를 선택하세요.",
    )

    col1, col2 = st.columns(2)
    with col1:
        accident_count = st.slider(
            "사고건수 (건)",
            min_value=0,
            max_value=5000,
            value=1300,
            step=10,
        )
    with col2:
        minor_injury_count = st.slider(
            "경상자수 (명)",
            min_value=0,
            max_value=5000,
            value=1350,
            step=10,
        )

    with st.expander("⚙️ 세부 부상자 변수 직접 조절 (선택 사항)"):
        use_custom_extra = st.checkbox(
            "중상자수·부상신고자수 직접 입력 (해제 시 모델 학습 평균값 자동 적용)",
            value=False,
        )
        if use_custom_extra:
            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                severe_injury_count = st.slider("중상자수 (명)", 0, 2000, 320, 10)
            with ex_col2:
                reported_injury_count = st.slider("부상신고자수 (명)", 0, 1500, 145, 5)
        else:
            severe_injury_count = None
            reported_injury_count = None


# ---------------------------------------------------------
# 5. 실시간 모델 예측 실행
# ---------------------------------------------------------
predicted_deaths = predict_casualties(
    orange_model=model,
    sido=selected_sido,
    accident_count=accident_count,
    minor_injury_count=minor_injury_count,
    severe_injury_count=severe_injury_count,
    reported_injury_count=reported_injury_count,
)


# ---------------------------------------------------------
# 6. 위험도 판정 및 대형 신호등 + 안전 수칙 카드 출력
# ---------------------------------------------------------
# 기준: 10명 미만 안전 / 10~20명 미만 주의 / 20명 이상 위험
if predicted_deaths < 10.0:
    level = "safe"
    badge_emoji = "🟢"
    level_label = "안전 (SAFE)"
    level_color = "#2ed573"
    level_desc = "예상 사망자 수가 낮은 수준입니다. 기본 안전 수칙을 준수하며 통학로 환경을 유지하세요."
    card_title = "🟢 어린이 보호구역 기본 안전 수칙 (안전 단계)"
    rules = [
        "<b>[운전자]</b> 어린이 보호구역 내 제한속도 <b>30km/h 이하</b> 준수 및 전방 주시 철저",
        "<b>[운전자]</b> 신호기가 없는 횡단보도 앞에서는 보행자 유무와 관계없이 <b>일시정지</b> 후 서행",
        "<b>[어린이·보행자]</b> 횡단보도를 건널 때는 <b>'멈추고 - 살피고 - 건너기'</b> 3원칙 실천",
        "<b>[어린이·보행자]</b> 보행 중 스마트폰·이어폰 사용 자제 및 밝은 색 옷·가방 착용",
    ]
elif predicted_deaths < 20.0:
    level = "caution"
    badge_emoji = "🟡"
    level_label = "주의 (CAUTION)"
    level_color = "#ffa502"
    level_desc = "사고 규모 대비 인명 피해 위험이 증가하고 있습니다. 등·하교 시간대 각별한 주의가 필요합니다."
    card_title = "🟡 어린이 보호구역 강화 안전 수칙 (주의 단계)"
    rules = [
        "<b>[운전자]</b> 주정차 전면 금지 구역 준수 (불법 주정차 차량은 어린이 사각지대 사고의 주원인)",
        "<b>[운전자]</b> 골목길·교차로 우회전 시 <b>완전 일시정지</b> 후 사각지대(A필러·차량 하단) 확인",
        "<b>[어린이·보행자]</b> 주정차된 차량 사이나 학원 차량 앞뒤로 갑자기 뛰어나오지 않기",
        "<b>[지역·학교]</b> 등·하교 집중 시간대 녹색어머니회·교통안전지도사 배치 및 옐로카펫 점검",
    ]
else:
    level = "danger"
    badge_emoji = "🔴"
    level_label = "위험 (DANGER)"
    level_color = "#ff4757"
    level_desc = "고위험 교통사고 다발 구간 수준입니다. 즉각적인 감속 운행과 집중 안전 관리가 요구됩니다."
    card_title = "🔴 어린이 보호구역 긴급 안전 수칙 (위험 단계)"
    rules = [
        "<b>[운전자]</b> 스쿨존 진입 전부터 <b>20~30km/h 이하</b> 감속 및 급가속·급제동 절대 금지",
        "<b>[운전자]</b> 어린이 통학버스 정차 및 점멸등 작동 시 차선 변경·추월 금지 및 일시정지",
        "<b>[어린이·보행자]</b> 우천·야간·시야 불량 시 반드시 보행자 전용 방호울타리 안쪽으로 통행",
        "<b>[지역·학교]</b> 과속단속 카메라·바닥신호등 작동 점검 및 위험 교차로 우회 통학로 안내",
    ]

st.subheader("2️⃣ 실시간 예측 결과 및 신호등 판정")

st.metric(
    label=f"📍 {selected_sido} 지역 예상 사망자 수",
    value=f"{predicted_deaths:.2f} 명",
    delta=f"{badge_emoji} {level_label}",
    delta_color="off",
)

# 대형 신호등 HTML 렌더링
green_cls = "bulb active-green" if level == "safe" else "bulb"
yellow_cls = "bulb active-yellow" if level == "caution" else "bulb"
red_cls = "bulb active-red" if level == "danger" else "bulb"

st.markdown(
    f"""
    <div class="traffic-box">
        <div class="lights-row">
            <div class="{green_cls}">🟢</div>
            <div class="{yellow_cls}">🟡</div>
            <div class="{red_cls}">🔴</div>
        </div>
        <div class="status-title" style="color: {level_color};">
            {badge_emoji} {level_label}
        </div>
        <div class="status-sub">{level_desc}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 어린이 보호구역 안전 수칙 카드 렌더링
rules_html = "".join(f"<li>{item}</li>" for item in rules)
st.markdown(
    f"""
    <div class="safety-card" style="border-left-color: {level_color};">
        <h4>{card_title}</h4>
        <ul>{rules_html}</ul>
    </div>
    """,
    unsafe_allow_html=True,
)
