import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. ضبط إعدادات الصفحة
st.set_page_config(
    page_title="AI Sentiment Analytics Engine",
    page_icon="🔮",
    layout="wide"
)

# 2. تحميل النماذج مباشرة لدعم التشغيل السحابي (Cloud Hosting)
@st.cache_resource
def load_models():
    try:
        nlp_model = joblib.load("nlp_sentiment_model.pkl")
        feature_model = joblib.load("feature_sentiment_model.pkl")
        scaler = joblib.load("scaler.pkl")
        num_cols = joblib.load("num_columns_list.pkl")
        return nlp_model, feature_model, scaler, num_cols
    except Exception as e:
        return None, None, None, None

nlp_model, feature_model, scaler, num_cols = load_models()

# 3. تصميم واجهة المستخدم
st.markdown("""
    <style>
    html, body, [class*="css"], p, span, label {
        color: #FFFFFF !important;
        font-weight: 500;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #F1F5F9 !important;
    }
    div[role="radiogroup"] {
        background: #1E293B !important;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #475569;
    }
    div[role="radiogroup"] label p {
        color: #F8FAFC !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        margin-bottom: 25px;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #A5B4FC, #E9D5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-sub {
        color: #CBD5E1 !important;
        font-size: 1.1rem;
    }
    .result-box {
        background: #1E293B;
        border-radius: 16px;
        padding: 20px;
        border-left: 6px solid #6366F1;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero-card">
        <div class="hero-title">🔮 AI Sentiment Prediction Platform</div>
        <div class="hero-sub">AI Analytics Cloud Engine & Machine Learning Pipelines</div>
    </div>
""", unsafe_allow_html=True)

col_input, col_display = st.columns([1, 1], gap="large")

label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

with col_input:
    st.subheader("⚙️ Configuration & Input")
    
    choice = st.radio(
        "Select Sentiment Analysis Mode:",
        ("NLP Text Classifier (Post Content)", "Numerical Engagement Features")
    )

    st.markdown("---")

    if choice == "NLP Text Classifier (Post Content)":
        user_text = st.text_area("✍️ Enter text/comment to analyze:", "The performance of this system is outstanding!", height=130)
        submit_btn = st.button("🚀 Analyze Text Sentiment", use_container_width=True)
    else:
        st.write("📊 Set Engagement Values:")
        user_inputs = []
        if num_cols is not None:
            c1, c2 = st.columns(2)
            for idx, col_name in enumerate(num_cols):
                with (c1 if idx % 2 == 0 else c2):
                    val = st.number_input(f"{col_name}", min_value=0, value=15)
                    user_inputs.append(float(val))
            submit_btn = st.button("🚀 Analyze Features", use_container_width=True)
        else:
            st.error("⚠️ Model files (.pkl) are missing in the repository.")
            submit_btn = None

with col_display:
    st.subheader("🎯 Real-Time Predictions & Metrics")
    
    if submit_btn and nlp_model is not None:
        try:
            if choice == "NLP Text Classifier (Post Content)":
                raw_pred = int(nlp_model.predict([user_text])[0])
                probs = nlp_model.predict_proba([user_text])[0]
            else:
                scaled_inputs = scaler.transform([user_inputs])
                raw_pred = int(feature_model.predict(scaled_inputs)[0])
                probs = feature_model.predict_proba(scaled_inputs)[0]

            pred_text = label_mapping.get(raw_pred, str(raw_pred))
            
            classes = getattr(nlp_model if choice.startswith("NLP") else feature_model, "classes_", range(len(probs)))
            confidence = {label_mapping.get(c, f"Class {c}"): float(p) for c, p in zip(classes, probs)}

            st.markdown(f"""
                <div class="result-box">
                    <h3 style="color: #94A3B8;">Prediction Result:</h3>
                    <h1 style="margin:0; color: #38BDF8;">{pred_text}</h1>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📈 Confidence Breakdown")
            for label, prob in confidence.items():
                st.write(f"**{label}**")
                st.progress(float(prob))

        except Exception as e:
            st.error(f"Error executing prediction: {e}")
    elif submit_btn:
        st.error("⚠️ Model files could not be loaded. Please ensure all .pkl files are committed to GitHub.")
    else:
        st.info("👈 Enter data and click Analyze to view real-time model outputs here.")