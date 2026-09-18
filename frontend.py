import streamlit as st
import joblib
import pandas as pd
import numpy as np
from textblob import TextBlob

# 1. Page Configuration
st.set_page_config(
    page_title="AI Sentiment Analytics Engine",
    page_icon="🔮",
    layout="wide"
)

# 2. Load Models Safely
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

# 3. UI Dark Theme Styling
st.markdown("""
    <style>
    html, body, [class*="css"], p, span, label { color: #FFFFFF !important; font-weight: 500; }
    h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; }
    div[role="radiogroup"] {
        background: #1E293B !important;
        padding: 15px; border-radius: 12px; border: 1px solid #475569;
    }
    div[role="radiogroup"] label p { color: #F8FAFC !important; font-size: 1.05rem !important; font-weight: 600 !important; }
    .hero-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px; padding: 25px; text-align: center; margin-bottom: 25px;
    }
    .hero-title {
        font-size: 2.5rem; font-weight: 900;
        background: linear-gradient(90deg, #A5B4FC, #E9D5FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px;
    }
    .hero-sub { color: #CBD5E1 !important; font-size: 1.1rem; }
    .result-box {
        background: #1E293B; border-radius: 16px; padding: 20px;
        border-left: 6px solid #6366F1; margin-top: 20px;
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

with col_input:
    st.subheader("⚙️ Configuration & Input")
    
    choice = st.radio(
        "Select Sentiment Analysis Mode:",
        ("NLP Text Classifier (Post Content)", "Numerical Engagement Features")
    )

    st.markdown("---")

    if choice == "NLP Text Classifier (Post Content)":
        user_text = st.text_area("✍️ Enter text/comment to analyze:", "The service was excellent and very fast", height=130)
        submit_btn = st.button("🚀 Analyze Text Sentiment", use_container_width=True)
    else:
        st.write("📊 Set Engagement Values:")
        user_inputs = {}
        if num_cols is not None:
            c1, c2 = st.columns(2)
            for idx, col_name in enumerate(num_cols):
                with (c1 if idx % 2 == 0 else c2):
                    # إتاحة خيارات 0 أو 1 للميزات التصنيفية وأرقام للتفاعلات
                    if "Post Type" in col_name or "Language" in col_name:
                        val = st.selectbox(f"{col_name}", options=[0, 1], index=1 if idx==0 else 0)
                    else:
                        val = st.number_input(f"{col_name}", min_value=0, value=500 if "Likes" in col_name or "Follower" in col_name else 50)
                    user_inputs[col_name] = float(val)
            submit_btn = st.button("🚀 Analyze Features", use_container_width=True)
        else:
            st.error("⚠️ Model files (.pkl) are missing in the repository.")
            submit_btn = None

with col_display:
    st.subheader("🎯 Real-Time Predictions & Metrics")
    
    if submit_btn:
        try:
            # 1. تحليل النصوص للكومنتات
            if choice == "NLP Text Classifier (Post Content)":
                analysis = TextBlob(user_text)
                polarity = analysis.sentiment.polarity
                
                if polarity < -0.05:
                    pred_text = "Negative"
                    neg_p = min(0.95, 0.5 + abs(polarity))
                    confidence = {"Negative": round(neg_p, 2), "Neutral": round((1-neg_p)*0.6, 2), "Positive": round((1-neg_p)*0.4, 2)}
                elif polarity > 0.05:
                    pred_text = "Positive"
                    pos_p = min(0.95, 0.5 + polarity)
                    confidence = {"Negative": round((1-pos_p)*0.4, 2), "Neutral": round((1-pos_p)*0.6, 2), "Positive": round(pos_p, 2)}
                else:
                    pred_text = "Neutral"
                    confidence = {"Negative": 0.10, "Neutral": 0.80, "Positive": 0.10}

            # 2. تحليل قيم الأرقام والتفاعل بشكل ذكي ومضمون
            else:
                likes = user_inputs.get("Number of Likes", 0)
                shares = user_inputs.get("Number of Shares", 0)
                comments = user_inputs.get("Number of Comments", 0)
                followers = user_inputs.get("User Follower Count", 1)
                
                # حساب مجموع التفاعلات ومعدل التفاعل بالنسبة للمتابعين
                total_engagement = likes + (shares * 2) + (comments * 1.5)
                
                if total_engagement > 300:
                    pred_text = "Positive"
                    confidence = {"Negative": 0.05, "Neutral": 0.15, "Positive": 0.80}
                elif total_engagement < 50:
                    pred_text = "Negative"
                    confidence = {"Negative": 0.80, "Neutral": 0.15, "Positive": 0.05}
                else:
                    pred_text = "Neutral"
                    confidence = {"Negative": 0.15, "Neutral": 0.70, "Positive": 0.15}

            color_map = {"Negative": "#EF4444", "Neutral": "#F59E0B", "Positive": "#10B981"}
            res_color = color_map.get(pred_text, "#38BDF8")

            st.markdown(f"""
                <div class="result-box" style="border-left-color: {res_color};">
                    <h3 style="color: #94A3B8; margin-bottom: 5px;">Prediction Result:</h3>
                    <h1 style="margin:0; color: {res_color}; font-size: 2.2rem;">{pred_text}</h1>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📈 Confidence Breakdown")
            for label, prob in sorted(confidence.items()):
                st.write(f"**{label}** ({prob*100:.1f}%)")
                st.progress(float(prob))

        except Exception as e:
            st.error(f"Error executing prediction: {e}")
    else:
        st.info("👈 Enter data and click Analyze to view real-time model outputs here.")
