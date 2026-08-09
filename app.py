import streamlit as st
import pandas as pd
import time
from src.inference import FraudInferenceEngine
from src.copilot import generate_fraud_triage_report

st.set_page_config(page_title="Enterprise Fraud System", layout="wide")
st.title("💳 Real-Time Credit Card Fraud Detection & GenAI Copilot")

# --- INITIALIZE INFERENCE ENGINE ---
@st.cache_resource
def get_engine():
    try:
        return FraudInferenceEngine()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run `python src/train.py` in your terminal to train and save the model first.")
        st.stop()

engine = get_engine()

# --- SIDEBAR CONTROL ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# --- DATA FOR STREAMING / TESTING ---
@st.cache_data
def get_test_data():
    return pd.read_csv("data/creditcard.csv").drop(['Class'], axis=1).sample(100, random_state=42)

test_data = get_test_data()

tab1, tab2 = st.tabs(["📡 Live Stream Simulator", "✍️ Manual Entry Tester"])

# ==================== TAB 1: STREAM SIMULATOR ====================
with tab1:
    if st.button("▶️ Start Streaming Simulation"):
        stream_spot = st.empty()
        alert_spot = st.empty()
        
        sample = test_data.sample(15)
        for i, (idx, row) in enumerate(sample.iterrows()):
            row_df = pd.DataFrame([row])
            pred, risk_score = engine.predict_transaction(row_df)
            
            status = "🟢 APPROVED" if pred == 0 else "🔴 FRAUD DETECTED"
            stream_spot.markdown(f"**Txn #{i+1} | ID: {idx} | Amount: ${row['Amount']:.2f} | Risk Score: {risk_score:.4f} -> {status}**")
            
            if pred == 1:
                with alert_spot.container():
                    st.error("⚠️ FRAUD DETECTED - Triggering GenAI Copilot...")
                    top_features = engine.get_shap_explanation(row_df)
                    report = generate_fraud_triage_report(row.to_dict(), top_features, api_key)
                    st.info(f"🤖 **GenAI Copilot Summary:**\n\n{report}")
                    time.sleep(4)
            else:
                time.sleep(0.5)

# ==================== TAB 2: MANUAL TESTING ====================
# ==================== TAB 2: MANUAL TESTING ====================
with tab2:
    st.markdown("### Test Custom Input Features")
    st.caption("Tip: In PCA datasets, fraud requires co-occurring anomalies across V17, V14, and V12.")
    
    with st.form("manual_entry"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount ($)", value=2500.0)
            v17 = st.slider("V17 (Primary PCA Anomaly)", -30.0, 30.0, -10.0)
            v14 = st.slider("V14 (Secondary PCA Anomaly)", -30.0, 30.0, -10.0)
            v12 = st.slider("V12 (Tertiary PCA Anomaly)", -30.0, 30.0, -5.0)
        with col2:
            time_val = st.number_input("Time", value=1000.0)
            v10 = st.slider("V10 (Device anomaly proxy)", -30.0, 30.0, -5.0)
            v11 = st.slider("V11 (Frequency anomaly proxy)", -30.0, 30.0, 5.0)
            v4 = st.slider("V4 (Velocity anomaly proxy)", -30.0, 30.0, 5.0)
            
        submitted = st.form_submit_button("Run Analysis")
        
    if submitted:
        sample_row = test_data.mean().to_dict()
        sample_row.update({
            'Amount': amount, 
            'Time': time_val, 
            'V17': v17,
            'V14': v14, 
            'V12': v12,
            'V10': v10, 
            'V11': v11,
            'V4': v4
        })
        input_df = pd.DataFrame([sample_row])
        
        pred, risk_score = engine.predict_transaction(input_df)
        if pred == 1:
            st.error(f"🔴 **FRAUD DETECTED** (Risk Score: {risk_score:.4f})")
            top_features = engine.get_shap_explanation(input_df)
            report = generate_fraud_triage_report(sample_row, top_features, api_key)
            st.info(f"🤖 **GenAI Copilot Summary:**\n\n{report}")
        else:
            st.success(f"🟢 **APPROVED** (Risk Score: {risk_score:.4f})")