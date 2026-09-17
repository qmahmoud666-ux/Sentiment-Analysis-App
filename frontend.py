import streamlit as st
import requests

# 1. ضبط إعدادات الصفحة
st.set_page_config(
    page_title="AI Sentiment Analytics Engine",
    page_icon="🔮",
    layout="wide"
)

# 2. إضافة CSS متقدم لتغيير تصميم الواجهة كلياً
st.markdown("""
    <style>
    /* تحسين البطاقة الرئيسية */
    .hero-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .hero-sub {
        color: #94A3B8;
        font-size: 1.1rem;
    }

    /* تحسين خيارات الراديو */
    div[role="radiogroup"] {
        background: #1E293B;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #334155;
    }

    /* تحسين كارت النتيجة */
    .result-box {
        background: #1E293B;
        border-radius: 16px;
        padding: 20px;
        border-left: 6px solid #6366F1;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# عرض الهيدر المخصص
st.markdown("""
    <div class="hero-card">
        <div class="hero-title">🔮 AI Sentiment Prediction Platform</div>
        <div class="hero-sub">Powered by FastAPI Back-End & Machine Learning Pipelines</div>
    </div>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

# تقسيم الشاشة إلى عمودين منظمين
col_input, col_display = st.columns([1, 1], gap="large")

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
        
        try:
            cols_response = requests.get(f"{API_URL}/get-columns").json()
            num_cols = cols_response["num_cols"]
            
            c1, c2 = st.columns(2)
            for idx, col_name in enumerate(num_cols):
                with (c1 if idx % 2 == 0 else c2):
                    val = st.number_input(f"{col_name}", min_value=0, value=15)
                    user_inputs.append(float(val))
            
            submit_btn = st.button("🚀 Analyze Features", use_container_width=True)
        except:
            st.error("⚠️ FastAPI is offline. Start `uvicorn backend:app --reload`")
            submit_btn = None

with col_display:
    st.subheader("🎯 Real-Time Predictions & Metrics")
    
    if 'submit_btn' in locals() and submit_btn:
        try:
            if choice == "NLP Text Classifier (Post Content)":
                res = requests.post(f"{API_URL}/predict-text", json={"text": user_text}).json()
            else:
                res = requests.post(f"{API_URL}/predict-features", json={"features": user_inputs}).json()
                
            pred_text = res["prediction"]
            pred_val = res["raw_pred"]
            confidence = res["confidence"]

            st.markdown(f"""
                <div class="result-box">
                    <h3>Prediction Result:</h3>
                    <h1 style="margin:0; color: #818CF8;">{pred_text}</h1>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📈 Confidence Breakdown")
            for label, prob in confidence.items():
                st.write(f"**{label}**")
                st.progress(float(prob))

        except Exception as e:
            st.error(f"Error communicating with Back-End API: {e}")
    else:
        st.info("👈 Enter data and click Analyze to view real-time model outputs here.")