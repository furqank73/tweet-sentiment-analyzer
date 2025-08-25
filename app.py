import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Load model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Custom CSS
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px;
        }
        .result-card {
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
        .positive {
            background-color: #e6ffed;
            color: #2e7d32;
            border: 2px solid #2e7d32;
        }
        .negative {
            background-color: #ffebee;
            color: #c62828;
            border: 2px solid #c62828;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("ℹ️ About this App")
st.sidebar.write("""
This **Sentiment Analysis App** uses a machine learning model 
trained with **1.6 million tweets** (Sentiment140 dataset) using **TF-IDF** features 
and a classifier to predict whether a given text expresses a 
**Positive** or **Negative** sentiment.
""")
st.sidebar.write("👨‍💻 Built by: [M Furqan Khan](https://www.linkedin.com/in/furqan-khan-256798268/)")


# Main title
st.markdown('<div class="main-title">💬 Sentiment Analysis App</div>', unsafe_allow_html=True)

# Example texts
user_text = st.text_area("✍️ Enter your text here:")

# Analyze button
if st.button("🔍 Analyze"):
    if user_text.strip():
        vec = vectorizer.transform([user_text])
        prediction = model.predict(vec)[0]

        # If model supports probabilities
        try:
            probs = model.predict_proba(vec)[0]
            confidence = np.max(probs) * 100
        except:
            probs, confidence = None, None

        if prediction in [4]:  # positive
            st.markdown('<div class="result-card positive">✅ Positive Sentiment</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-card negative">❌ Negative Sentiment</div>', unsafe_allow_html=True)

        # Confidence gauge chart
        if probs is not None:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Please enter some text first")
