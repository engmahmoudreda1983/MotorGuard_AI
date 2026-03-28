# ==============================================================================
# © 2026 MotorGuard AI | Predictive Maintenance & Fault Diagnostics
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
# 1. PAGE SETUP
# =========================================
st.set_page_config(page_title="MotorGuard AI Pro", page_icon="⚙️", layout="wide")

if 'inspection_log' not in st.session_state:
    st.session_state['inspection_log'] = pd.DataFrame(columns=['Timestamp', 'Type', 'Overall Risk', 'RPM', 'Diagnosis', 'Status'])

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
        st.error(f"⚠️ Error loading AI files: scaler.pkl, motorguard_model.pkl. Details: {e}")
        return None, None

scaler, model = load_ai_assets()

# --- Feature Importance Data ---
data_pattern_importances = pd.DataFrame({
    'Feature': ['Torque', 'RPM', 'Wear', 'AirTemp', 'ProcTemp', 'Type'],
    'Impact': [0.3111, 0.2973, 0.2063, 0.1020, 0.0658, 0.0174]
})
data_pattern_importances = data_pattern_importances.sort_values(by='Impact', ascending=True)

# =========================================
# 3. AUTHENTICATION (SIMPLIFIED)
# =========================================
def check_password():
    if st.session_state["authenticated"]:
        return True
        
    st.markdown("""
    <div style="background-color:#F5F5F5; padding:20px; border-radius:15px; border-bottom: 5px solid #DD4B39; text-align: center;">
        <h1 style="color:#1E1E1E; margin:0;">MotorGuard <span style="color:#DD4B39;">AI</span></h1>
        <p style="color:#888888;">Secure Industrial Login</p>
    </div>
    """, unsafe_allow_html=True)
    
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if user == "user" and password == "pass":
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ Incorrect credentials")
    return False

def create_industrial_gauge(value, title, unit, max_val, thresholds):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': f"{title} ({unit})", 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, max_val]},
            'bar': {'color': "#1E1E1E"},
            'steps': [
                {'range': [0, thresholds['warn']], 'color': "#00A65A"},
                {'range': [thresholds['warn'], thresholds['danger']], 'color': "#f39c12"},
                {'range': [thresholds['danger'], max_val], 'color': "#DD4B39"}
            ]}))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# =========================================
# 4. MAIN APP
# =========================================
if not check_password():
    st.stop()

# Header
st.markdown("""
<div style="background-color:#1E1E1E; padding:15px; border-radius:10px; border-left: 8px solid #DD4B39;">
    <h1 style="color:white; margin:0;">⚙️ MotorGuard AI <span style="color:#DD4B39;">Pro</span></h1>
    <p style="color:#B0B0B0; margin:0;">Industrial Predictive Maintenance V1.0 | © 2026 Eng. Mahmoud</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ SCADA Inputs")
    motor_type = st.selectbox("Motor Type", ['L (Low)', 'M (Medium)', 'H (High)'], index=1)
    air_temp = st.number_input("Air Temp [K]", 290.0, 310.0, 298.0)
    proc_temp = st.number_input("Process Temp [K]", 300.0, 320.0, 308.0)
    rpm = st.slider("Speed [RPM]", 1100, 2900, 1500)
    torque = st.slider("Torque [Nm]", 10.0, 80.0, 40.0)
    wear = st.slider("Tool Wear [min]", 0, 260, 50)
    analyze_btn = st.button("🔍 Run Analysis", type="primary", use_container_width=True)

# Analysis Logic
if analyze_btn and scaler and model:
    t_map = {'L (Low)': 0, 'M (Medium)': 1, 'H (High)': 2}
    raw = pd.DataFrame([[t_map[motor_type], air_temp, proc_temp, rpm, torque, wear]], 
                      columns=['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
    
    scaled = raw.copy()
    scaled.iloc[:, 1:] = scaler.transform(raw.iloc[:, 1:])
    
    diag = model.predict(scaled)
    prob = model.predict_proba(scaled)
    risk = sum(prob[1:]) * 100

    # Results Banner
    color = "#00A65A" if diag == 'Normal' else "#DD4B39"
    st.markdown(f"""<div style="background-color:{color}10; padding:15px; border-radius:10px; border:2px solid {color}; margin-top:10px;">
        <h3 style="color:{color};">Status: {diag.upper()}</h3>
    </div>""", unsafe_allow_html=True)

    # Gauges Row - Fixed using integer for columns
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.plotly_chart(create_industrial_gauge(wear, "Wear", "min", 260, {'warn':150, 'danger':200, 'critical':240}), use_container_width=True)
    with c2: st.plotly_chart(create_industrial_gauge(air_temp, "Air T", "K", 310, {'warn':301, 'danger':303, 'critical':305}), use_container_width=True)
    with c3: st.plotly_chart(create_industrial_gauge(proc_temp, "Proc T", "K", 320, {'warn':310, 'danger':315, 'critical':318}), use_container_width=True)
    with c4: st.plotly_chart(create_industrial_gauge(risk, "Risk", "%", 100, {'warn':25, 'danger':50, 'critical':80}), use_container_width=True)

    # History Log
    new_entry = pd.DataFrame([[datetime.now().strftime("%H:%M:%S"), motor_type, f"{risk:.1f}%", rpm, diag, "Safe" if diag=="Normal" else "Critical"]], 
                             columns=st.session_state['inspection_log'].columns)
    st.session_state['inspection_log'] = pd.concat([new_entry, st.session_state['inspection_log']])

# History Table
st.markdown("---")
st.subheader("📋 Inspection History")
st.dataframe(st.session_state['inspection_log'], use_container_width=True)

# Footer
st.markdown("<hr><div style='text-align:center; color:#888;'>Developed by Eng. Mahmoud © 2026</div>", unsafe_allow_html=True)