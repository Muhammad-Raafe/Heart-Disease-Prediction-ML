import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 15px;
        border: 1px solid #e74c3c;
        margin-bottom: 25px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2d2d44;
        text-align: center;
        margin: 5px;
    }
    .section-header {
        color: #e74c3c;
        font-size: 18px;
        font-weight: bold;
        border-bottom: 2px solid #e74c3c;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #c0392b, #a93226);
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        color: #a0a0b0 !important;
        font-size: 13px;
    }
    .result-high {
        background: linear-gradient(135deg, #2d1b1b, #3d1a1a);
        border: 2px solid #e74c3c;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #1b2d1b, #1a3d1a);
        border: 2px solid #2ecc71;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def train_model():
    df = pd.read_csv("heart_disease_uci.csv")

    df["restecg"] = df["restecg"].fillna(df["restecg"].mode()[0])
    df["chol"] = df["chol"].fillna(df["chol"].median())
    df["trestbps"] = df["trestbps"].fillna(df["trestbps"].median())
    df["thalch"] = df["thalch"].fillna(df["thalch"].median())
    df["exang"] = df["exang"].fillna(df["exang"].mode()[0])
    df["oldpeak"] = df["oldpeak"].fillna(df["oldpeak"].median())
    df["slope"] = df["slope"].fillna(df["slope"].mode()[0])
    df["fbs"] = df["fbs"].fillna(df["fbs"].mode()[0])
    df = df.drop(["thal", "ca"], axis=1)

    df = pd.get_dummies(df, columns=["dataset", "slope", "restecg", "cp"], drop_first=True)

    le = LabelEncoder()
    df["sex"] = le.fit_transform(df["sex"])
    df["fbs"] = le.fit_transform(df["fbs"])
    df["exang"] = le.fit_transform(df["exang"])
    df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

    x = df.drop("num", axis=1)
    y = df["num"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, y_train)

    acc = accuracy_score(y_test, model.predict(x_test))
    return model, scaler, x.columns.tolist(), acc


model, scaler, feature_cols, acc = train_model()

# --- Header ---
st.markdown(f"""
<div class="main-header">
    <h1>❤️ Heart Disease Risk Predictor</h1>
    <p style="color:#a0a0b0;">AI-powered cardiovascular risk assessment using Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

# --- Top Metrics ---
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="color:#e74c3c;">{acc:.0%}</h2>
        <p style="color:#a0a0b0;">Model Accuracy</p>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="color:#3498db;">920+</h2>
        <p style="color:#a0a0b0;">Patients Trained On</p>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="color:#2ecc71;">Logistic Regression</h2>
        <p style="color:#a0a0b0;">ML Algorithm</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Input Form ---
st.markdown('<p class="section-header">🧾 Patient Information</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)

with col2:
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["FALSE", "TRUE"])
    restecg = st.selectbox("Resting ECG Result", ["normal", "st-t abnormality", "lv hypertrophy"])
    thalch = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)

with col3:
    exang = st.selectbox("Exercise Induced Angina", ["FALSE", "TRUE"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of ST Segment", ["upsloping", "flat", "downsloping"])
    dataset = st.selectbox("Patient Location", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])

st.markdown("<br>", unsafe_allow_html=True)

# --- Predict Button ---
if st.button("🔍 Analyze Heart Disease Risk"):

    input_dict = {col: [0] for col in feature_cols}
    input_dict["age"] = [age]
    input_dict["trestbps"] = [trestbps]
    input_dict["chol"] = [chol]
    input_dict["thalch"] = [thalch]
    input_dict["oldpeak"] = [oldpeak]
    input_dict["sex"] = [1 if sex == "Male" else 0]
    input_dict["fbs"] = [1 if fbs == "TRUE" else 0]
    input_dict["exang"] = [1 if exang == "TRUE" else 0]

    for col in feature_cols:
        if "dataset_" in col:
            input_dict[col] = [1 if col == f"dataset_{dataset}" else 0]
        if "slope_" in col:
            input_dict[col] = [1 if col == f"slope_{slope}" else 0]
        if "restecg_" in col:
            input_dict[col] = [1 if col == f"restecg_{restecg}" else 0]
        if "cp_" in col:
            input_dict[col] = [1 if col == f"cp_{cp}" else 0]

    input_df = pd.DataFrame(input_dict)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">📊 Analysis Results</p>', unsafe_allow_html=True)

    res_col, gauge_col = st.columns([1, 1])

    with res_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-high">
                <h1>⚠️</h1>
                <h2 style="color:#e74c3c;">High Risk Detected</h2>
                <h3 style="color:white;">{probability:.0%} probability of heart disease</h3>
                <p style="color:#a0a0b0;">Please consult a cardiologist immediately.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <h1>✅</h1>
                <h2 style="color:#2ecc71;">Low Risk</h2>
                <h3 style="color:white;">{probability:.0%} probability of heart disease</h3>
                <p style="color:#a0a0b0;">Maintain a healthy lifestyle.</p>
            </div>""", unsafe_allow_html=True)

    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Risk Score", "font": {"color": "white", "size": 20}},
            number={"suffix": "%", "font": {"color": "white", "size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": "#e74c3c" if probability > 0.5 else "#2ecc71"},
                "bgcolor": "#1a1a2e",
                "steps": [
                    {"range": [0, 40], "color": "#1a3d1a"},
                    {"range": [40, 70], "color": "#3d3d1a"},
                    {"range": [70, 100], "color": "#3d1a1a"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor="#0e1117",
            font={"color": "white"},
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
