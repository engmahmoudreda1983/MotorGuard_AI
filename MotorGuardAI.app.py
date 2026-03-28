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
st.set_page_config(page_title="MotorGuard AI Pro", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #DD4B39;
        color: white;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

if 'inspection_log' not in st.session_state:
    st.session_state['inspection_log'] = pd.DataFrame(columns=['Timestamp', 'Motor Type', 'Speed (RPM)', 'Overall Risk', 'Diagnosis', 'Status'])

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
        st.error(f"⚠️ Error loading AI files: {e}")
        return None, None

scaler, model = load_ai_assets()

# =========================================
# 3. AUTHENTICATION & GAUGES
# =========================================
def check_password():
    if st.session_state["authenticated"]: return True
    col1, col2, col3 = st.columns((1.5, 2.0, 1.5)) 
    with col2: 
        st.markdown("""
        <div style="background-color:#F8F9FA; padding:30px; border-radius:15px; border-top: 5px solid #DD4B39; text-align: center; margin-top: 50px;">
            <h2 style="color:#1E1E1E;">MotorGuard <span style="color:#DD4B39;">AI</span></h2>
            <p style="color:#666;">Secure Industrial Portal</p>
        </div><br>
        """, unsafe_allow_html=True)
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login securely", use_container_width=True, type="primary"):
            if user == "admin" and password == "DBA2026":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Incorrect Username or Password")
    return False

def create_gauge(value, title, color, max_val=100):
    val = float(value)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        number = {'suffix': "%", 'font': {'size': 20, 'color': color}},
        title = {'text': title, 'font': {'size': 13}},
        gauge = {
            'axis': {'range': [0, max_val]},
            'bar': {'color': color},
            'steps': [
                {'range':, 'color': '#E8F5E9'},
                {'range':, 'color': '#FEF9E7'},
                {'range':, 'color': '#FDEDEC'}
            ],
            'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': val}
        }))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# =========================================
# 4. MAIN APP
# =========================================
if not check_password(): st.stop()

st.markdown('<div class="main-header"><h2>⚙️ MotorGuard AI <span style="color:#DD4B39;">Pro</span></h2><p style="color:#CCC;">Industrial Predictive Maintenance Dashboard V2.0</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎛️ SCADA Telemetry")
    motor_type = st.selectbox("Motor Quality", ['L (Low)', 'M (Medium)', 'H (High)'], index=1)
    st.markdown("---")
    air_temp = st.number_input("Air Temp [K]", 290.0, 310.0, 298.0, step=0.1)
    process_temp = st.number_input("Process Temp [K]", 300.0, 320.0, 308.0, step=0.1)
    st.markdown("---")
    rpm = st.slider("Speed [RPM]", 1100, 2900, 1500, step=10)
    torque = st.slider("Torque [Nm]", 10.0, 80.0, 40.0, step=0.5)
    tool_wear = st.slider("Tool Wear [min]", 0, 260, 50, step=1)
    st.markdown("---")
    analyze_btn = st.button("🔍 Run Full Diagnostics", use_container_width=True, type="primary")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

if analyze_btn and scaler and model:
    # 1. AI Prediction (Overall Risk)
    t_map = {'L (Low)': 0, 'M (Medium)': 1, 'H (High)': 2}
    raw_df = pd.DataFrame([[t_map[motor_type], air_temp, process_temp, rpm, torque, tool_wear]], 
                          columns=['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
    
    scaled = raw_df.copy()
    scaled.iloc[:, 1:] = scaler.transform(raw_df.iloc[:, 1:])
    
    prediction = model.predict(scaled)
    probs = model.predict_proba(scaled)
    
    # Calculate Overall Risk from AI
    overall_risk = float(100.0 - (probs * 100))
    diag_label = "NORMAL" if overall_risk < 40 else "FAILURE PREDICTED"
    
    # 2. Expert System (Physical Equations for specific gauges so they MOVE!)
    # HDF (Heat Failure): Driven by temperature difference
    temp_diff = process_temp - air_temp
    p_hdf = max(0.0, min(100.0, (temp_diff - 8.0) * 12.5)) if process_temp > 308 else 0.0
    
    # TWF (Wear Failure): Driven by tool wear
    p_twf = max(0.0, min(100.0, (tool_wear / 240.0) * 100.0))
    
    # OSF (Overstrain): Driven by Torque and Wear
    p_osf = max(0.0, min(100.0, ((torque * tool_wear) / 16000.0) * 100.0))
    
    # PWF (Power): Driven by Power (Torque * RPM)
    power = torque * rpm
    p_pwf = max(0.0, min(100.0, abs(power - 60000) / 20000.0 * 100.0)) if power > 70000 or power < 50000 else 0.0

    # 3. Dynamic UI Update
    if overall_risk < 40:
        c_bg, c_border, c_text, status = "#E8F5E9", "#00A65A", "✅ SYSTEM STATUS: NORMAL", "Safe"
    elif overall_risk < 75:
        c_bg, c_border, c_text, status = "#FEF9E7", "#F39C12", f"⚠️ WARNING: RISK AT {overall_risk:.1f}%", "Warning"
    else:
        c_bg, c_border, c_text, status = "#FDEDEC", "#DD4B39", "🚨 DANGER: IMMINENT FAILURE", "Critical"

    st.markdown(f'<div style="background-color:{c_bg}; border:2px solid {c_border}; padding:15px; border-radius:10px; text-align:center;"><h2 style="color:{c_border}; margin:0;">{c_text}</h2></div>', unsafe_allow_html=True)
    st.markdown("---")

    # Gauges that actually MOVE!
    st.markdown("### 📊 Failure Probability Analysis (Real-time)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.plotly_chart(create_gauge(overall_risk, "OVERALL RISK", "#1E1E1E"), use_container_width=True)
    with c2: st.plotly_chart(create_gauge(p_hdf, "Heat (HDF)", "#E67E22"), use_container_width=True)
    with c3: st.plotly_chart(create_gauge(p_osf, "Overstrain (OSF)", "#9B59B6"), use_container_width=True)
    with c4: st.plotly_chart(create_gauge(p_pwf, "Power (PWF)", "#E74C3C"), use_container_width=True)
    with c5: st.plotly_chart(create_gauge(p_twf, "Wear (TWF)", "#34495E"), use_container_width=True)

    # 4. Dynamic Bar Chart (Moves with Sliders!)
    st.markdown("---")
    st.markdown("### 📈 Live Sensor Stress Impact")
    
    impact_data = pd.DataFrame({
        'Sensor Parameter': ['Process Temp', 'Air Temp', 'Tool Wear', 'RPM', 'Torque'],
        'Stress Level (%)': [p_hdf, p_hdf*0.8, p_twf, p_pwf*0.6, p_osf*0.9] # Linked to actual physics
    })
    # Ensure minimum visibility
    impact_data['Stress Level (%)'] = impact_data['Stress Level (%)'] + 5.0 
    impact_data = impact_data.sort_values(by='Stress Level (%)', ascending=True)

    fig_bar = px.bar(impact_data, x='Stress Level (%)', y='Sensor Parameter', orientation='h', color_discrete_sequence=['#3498DB'])
    fig_bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Log Update
    new_log = pd.DataFrame([[datetime.now().strftime("%H:%M:%S"), motor_type, rpm, f"{overall_risk:.1f}%", diag_label, status]], columns=st.session_state.inspection_log.columns)
    st.session_state.inspection_log = pd.concat([new_log, st.session_state.inspection_log], ignore_index=True)

st.markdown("---")
st.markdown("### 📋 Automated Inspection Log")
st.dataframe(st.session_state.inspection_log, use_container_width=True, hide_index=True)
st.markdown("<hr><center><p style='color:#888;'>MotorGuard AI Pro V2.0 | © 2026 Eng. Mahmoud</p></center>", unsafe_allow_html=True)