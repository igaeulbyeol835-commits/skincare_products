import streamlit as st
import pandas as pd
import plotly.express as px
import os
import ast

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="화장품 제형 분류 실습실",
    page_icon="🧴",
    layout="wide"
)

# 세션 상태 초기화
if 'model_pipeline' not in st.session_state:
    st.session_state['model_pipeline'] = None
if 'is_trained' not in st.session_state:
    st.session_state['is_trained'] = False
if 'eval_metrics' not in st.session_state:
    st.session_state['eval_metrics'] = {}

# 데이터 로드 함수
@st.cache_data
def load_data(uploaded_file=None):
    file_path = "skincare_products_clean.csv"
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")
            return None
    elif os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            st.error(f"기본 파일 읽기 오류: {e}")
            return None
    return df

# 성분 텍스트 전처리 함수
def preprocess_ingredients_text(text):
    if pd.isna(text):
        return ""
    text_str = str(text)
    if text_str.startswith("[") and text_str.endswith("]"):
        try:
            parsed = ast.literal_eval(text_str)
            if isinstance(parsed, list):
                return " ".join([str(item).replace(" ", "_") for item in parsed])
        except Exception:
            pass
    return text_str.replace("[", "").replace("]", "").replace("'", "").replace('"', "")

# 사이드바 메뉴
st.sidebar.title("🧴 메뉴")
menu = st.sidebar.radio(
    "이동할 화면:",
    [
        "1. 데이터 요약 및 탐색",
        "2. 머신러닝 모델 학습",
        "3. 새로운 성분 조합 예측"
    ]
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
df_raw = load_data(uploaded_file)

# 메인 타이틀
st.title("🧴 화장품 제형 분류 실습실")
st.markdown("---")

# -----------------------------------------------------------------------------
# 1. 데이터 요약 및 탐색
# -----------------------------------------------------------------------------
if menu == "1. 데이터 요약 및 탐색":
    st.subheader("📊 데이터 요약 및 구조")
    
    if df_raw is None:
        st.warning("⚠️ 'skincare_products_clean.csv' 파일을 찾을 수 없습니다. 사이드바에서 파일을 업로드해 주세요.")
    else:
        # 데이터 요약 카드
        total_count = len(df_raw)
        type_count = df_raw['product_type'].nunique() if 'product_type' in df_raw.columns else 0
        
        moisturiser_count = 0
        if 'product_type' in df_raw.columns:
            moisturiser_count = (df_raw['product_type'].astype(str).str.strip().str.lower() == 'moisturiser').sum()
        moisturiser_ratio = (moisturiser_count / total_count * 100) if total_count > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("전체 데이터 개수", f"{total_count:,} 개")
        c2.metric("제품 유형 개수", f"{type_count:,} 개")
        c3.metric("보습제 개수", f"{moisturiser_count:,} 개")
        c4.metric("보습제 비율", f"{moisturiser_ratio:.1f}%")

        st.markdown("---")
        st.markdown("#### 컬럼별 구조 및 결측치 확인")
        struct_list = []
        for col in df_raw.columns:
            struct_list.append({
                "컬럼명": col,
                "데이터 유형": str(df_raw[col].dtype),
                "결측치 개수": int(df_raw[col].isna().sum()),
                "고유값 개수": int(df_raw[col].nunique())
            })
        st.dataframe(pd.DataFrame(struct_list), use_container_width=True)

        st.markdown("#### 데이터 미리보기 (상위 5행)")
        st.dataframe(df_raw.head(5), use_container_width=True)

        if 'product_type' in df_raw.columns:
            st.markdown("#### 제품 유형별 분포")
            type_df = df_raw['product_type'].value_counts().reset_index()
            type_df.columns = ['product_type', 'count']
            fig = px.bar(type_df, x='product_type', y='count', title="제품 유형별 데이터 개수")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 2. 머신러닝 모델 학습
# -----------------------------------------------------------------------------
elif menu == "2. 머신러닝 모델 학습":
    st.subheader("🤖 머신러닝 모델 학습 및 평가")
    
    if df_raw is None:
        st.warning("⚠️ 데이터가 로드되지 않았습니다. 파일을 먼저 확인해 주세요.")
    else:
        req_cols = ['clean_ingreds', 'product_type']
        if not all(c in df_raw.columns for c in req_cols):
            st.error(f"필수 컬럼({req_cols})이 누락되었습니다.")
        else:
            df_ml = df_raw.dropna(subset=['clean_ingreds', 'product_type']).copy()
            df_ml['processed_text'] = df_ml['clean_ingreds'].apply(preprocess_ingredients_text)
            df_ml['target'] = df_ml['product_type'].astype(str).str.strip().str.lower().apply(
                lambda x: 1 if x == 'moisturiser' else 0
            )

            pos_count = (df_ml['target'] == 1).sum()
            neg_count = (df_ml['target'] == 0).sum()

            st.write(f"- 학습용 유효 데이터: {len(df_ml)}건 (보습제: {pos_count}건, 기타: {neg_count}건)")

            if len(df_ml) < 10 or pos_count == 0 or neg_count == 0:
                st.error("❌ 학습에 필요한 데이터가 부족하거나 특정 클래스가 존재하지 않습니다.")
            else:
                if st.button("모델 학습 실행", use_container_width=True):
                    X = df_ml['processed_text']
                    y = df_ml['target']

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )

                    pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(max_features=1000)),
                        ('clf', LogisticRegression(random_state=42))
                    ])

                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    st.session_state['model_pipeline'] = pipeline
                    st.session_state['is_trained'] = True
                    st.session_state['eval_metrics'] = {
                        'acc': accuracy_score(y_test, y_pred),
                        'prec': precision_score(y_test, y_pred, zero_division=0),
                        'rec': recall_score(y_test, y_pred, zero_division=0),
                        'f1': f1_score(y_test, y_pred, zero_division=0),
                        'cm': confusion_matrix(y_test, y_pred)
                    }
                    st.success("모델 학습이 완료되었습니다.")

            if st.session_state['is_trained']:
                m = st.session_state['eval_metrics']
                st.markdown("---")
                st.markdown("#### 모델 평가 지표")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("정확도 (Accuracy)", f"{m['acc']*100:.2f}%")
                col2.metric("정밀도 (Precision)", f"{m['prec']*100:.2f}%")
                col3.metric("재현율 (Recall)", f"{m['rec']*100:.2f}%")
                col4.metric("F1-Score", f"{m['f1']:.3f}")

                st.markdown("#### 혼동 행렬 (Confusion Matrix)")
                fig_cm = px.imshow(
                    m['cm'],
                    text_auto=True,
                    x=['기타(0)', '보습제(1)'],
                    y=['기타(0)', '보습제(1)'],
                    labels=dict(x="예측값", y="실제값")
                )
                st.plotly_chart(fig_cm, use_container_width=True)

# -----------------------------------------------------------------------------
# 3. 새로운 성분 조합 예측
# -----------------------------------------------------------------------------
elif menu == "3. 새로운 성분 조합 예측":
    st.subheader("🔮 성분 조합 기반 제품 유형 예측")
    
    if not st.session_state['is_trained'] or st.session_state['model_pipeline'] is None:
        st.warning("⚠️ 학습된 모델이 없습니다. [2. 머신러닝 모델 학습] 메뉴에서 먼저 모델을 학습시켜 주세요.")
    else:
        user_input = st.text_area("성분 목록을 입력하세요 (쉼표 구분)", placeholder="예: glycerin, water, cetyl alcohol, hyaluronic acid")

        if st.button("예측하기", use_container_width=True):
            if not user_input.strip():
                st.error("성분 정보를 입력해주세요.")
            else:
                processed = preprocess_ingredients_text(user_input)
                pipeline = st.session_state['model_pipeline']
                
                pred = pipeline.predict([processed])[0]
                proba = pipeline.predict_proba([processed])[0]

                st.markdown("---")
                if pred == 1:
                    st.success(f"**예측 결과: 보습제 (Moisturiser)** (확률: {proba[1]*100:.2f}%)")
                else:
                    st.info(f"**예측 결과: 기타 화장품 (Other)** (확률: {proba[0]*100:.2f}%)")

