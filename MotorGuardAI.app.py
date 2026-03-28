# ==============================================================================
# © 2026 MotorGuard AI | Advanced Predictive Maintenance
# Developed by: Eng. Mahmoud
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime

# =========================================
# 1. PAGE SETUP & STYLING
# =========================================
st.set_page_config(page_title="MotorGuard AI Pro", page_icon="⚙️", layout="wide")

# Custom CSS for Login Centering
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 25px; }
    .main-header { background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 10px solid #DD4B39; color: white; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

if 'inspection_log' not in st.session_state:
    st.session_state['inspection_log'] = pd.DataFrame(columns=['Timestamp', 'Type', 'Overall Risk', 'Diagnosis', 'Status'])

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# =========================================
# 2. AI MODELS LOADING
# =========================================
@st.cache_resource
def load_ai_assets():
    try:
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('motorguard_model.pkl')
        return scaler, model
    except Exception as e:
        st.error(f"⚠️ Missing AI Assets: {e}")
        return None, None

scaler, model = load_ai_assets()

# =========================================
# 3. AUTHENTICATION (Centered & Adjusted)
# =========================================
def check_password():
    if st.session_state["authenticated"]:
        return True
        
    # Centering the Login Box
    _, col_mid, _ = st.columns([1, 1.5, 1])
    with col_mid:
        st.markdown("""
            <div style="background-color:#F8F9FA; padding:30px; border-radius:15px; border-top: 5px solid #DD4B39; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h1 style="color:#1E1E1E; margin-bottom:0;">MotorGuard <span style="color:#DD4B39;">AI</span></h1>
                <p style="color:#666;">Secure Engineering Portal</p>
            </div>
        """, unsafe_allow_html=True)
        
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Access Dashboard", use_container_width=True, type="primary"):
            if user == "admin" and password == "DBA2026":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid Credentials")
    return False

def logout():
    st.session_state["authenticated"] = False
    st.rerun()

# =========================================
# 4. GAUGE CREATOR
# =========================================
def create_gauge(value, title, color="#DD4B39", is_pct=True):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [0, 100 if is_pct else 260]},
            'bar': {'color': color},
            'steps': [{'range': [0, 50 if is_pct else 150], 'color': "#F5F5F5"}],
            'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': value}
        }))
    fig.update_layout(height=180, margin=dict(l=15, r=15, t=40, b=10))
    return fig

# =========================================
# 5. MAIN APPLICATION
# =========================================
if not check_password():
    st.stop()

# Header
st.markdown('<div class="main-header"><h2>⚙️ MotorGuard AI - Predictive Analytics Pro</h2><p>Real-time Industrial Failure Detection | © 2026 Eng. Mahmoud</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ SCADA Control")
    m_type = st.selectbox("Motor Grade", ['L (Low)', 'M (Medium)', 'H (High)'], index=1)
    air_t = st.number_input("Air Temp [K]", 290.0, 310.0, 298.0)
    proc_t = st.number_input("Process Temp [K]", 300.0, 320.0, 308.0)
    rpm = st.slider("Speed [RPM]", 1100, 2900, 1500)
    torque = st.slider("Torque [Nm]", 10.0, 80.0, 40.0)
    wear = st.slider("Tool Wear [min]", 0, 260, 50)
    
    analyze = st.button("🔍 Run Full Diagnostics", type="primary", use_container_width=True)
    st.markdown("---")
    if st.button("🚪 Logout"): logout()

# Analysis Logic
if analyze and scaler and model:
    # Data Processing
    t_map = {'L (Low)': 0, 'M (Medium)': 1, 'H (High)': 2}
    raw = pd.DataFrame([[t_map[m_type], air_t, proc_t, rpm, torque, wear]], 
                      columns=['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
    
    scaled = raw.copy()
    scaled.iloc[:, 1:] = scaler.transform(raw.iloc[:, 1:])
    
    # AI Predictions
    diag_raw = model.predict(scaled)
    diag = str(diag_raw)
    probs = model.predict_proba(scaled)
    
    # Probabilities Mapping (Based on common LSS Failure classes)
    # Class order usually: [Normal, HDF, OSF, PWF, TWF, RNF]
    p_normal = probs * 100
    p_heat = probs * 100 if len(probs) > 1 else 0
    p_overstrain = probs * 100 if len(probs) > 2 else 0
    p_power = probs * 100 if len(probs) > 3 else 0
    p_wear = probs * 100 if len(probs) > 4 else 0
    
    overall_risk = (1 - probs) * 100

    # Results Section
    st.subheader("🖥️ Diagnostic Summary")
    banner_color = "#00A65A" if diag == 'Normal' else "#DD4B39"
    st.markdown(f'<div style="background-color:{banner_color}; color:white; padding:15px; border-radius:10px; text-align:center;"><h3>SYSTEM STATUS: {diag.upper()}</h3></div>', unsafe_allow_html=True)

    # Risk Gauges Row
    st.markdown("### ⚠️ Failure Risk Analysis")
    g1, g2, g3, g4, g5 = st.columns(5)
    with g1: st.plotly_chart(create_gauge(overall_risk, "OVERALL RISK", "#000000"), use_container_width=True)
    with g2: st.plotly_chart(create_gauge(p_heat, "Heat (HDF)", "#FF5733"), use_container_width=True)
    with g3: st.plotly_chart(create_gauge(p_overstrain, "Overstrain (OSF)", "#C70039"), use_container_width=True)
    with g4: st.plotly_chart(create_gauge(p_power, "Power (PWF)", "#900C3F"), use_container_width=True)
    with g5: st.plotly_chart(create_gauge(p_wear, "Wear (TWF)", "#581845"), use_container_width=True)

    # Logging
    new_log = pd.DataFrame([[datetime.now().strftime("%H:%M:%S"), m_type, f"{overall_risk:.1f}%", diag, "Critical" if diag != 'Normal' else "Safe"]], 
                          columns=st.session_state['inspection_log'].columns)
    st.session_state['inspection_log'] = pd.concat([new_log, st.session_state['inspection_log']], ignore_index=True)

# History Log
st.markdown("---")
st.subheader("📋 Inspection History")
st.dataframe(st.session_state['inspection_log'], use_container_width=True)

# Footer
st.markdown("<br><hr><center><p style='color:#888;'>MotorGuard AI Engine Pro V2.0 | Secured Industrial Asset Monitoring</p></center>", unsafe_allow_html=True)