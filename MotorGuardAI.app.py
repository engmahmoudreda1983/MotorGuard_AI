# ==============================================================================
# © 2026 MotorGuard AI | Predictive Maintenance & Fault Diagnostics
# Developed by: Eng. Mahmoud
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import datetime

# =========================================
# 1. PAGE SETUP & THEME
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
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .status-safe { background-color: #E8F5E9; border: 2px solid #00A65A; padding: 15px; border-radius: 10px; text-align: center; }
    .status-critical { background-color: #FDEDEC; border: 2px solid #DD4B39; padding: 15px; border-radius: 10px; text-align: center; }
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
        st.error(f"⚠️ Error loading AI files: scaler.pkl, motorguard_model.pkl. Details: {e}")
        return None, None

scaler, model = load_ai_assets()

# =========================================
# 3. HELPER FUNCTIONS & AUTHENTICATION
# =========================================
def check_password():
    if st.session_state["authenticated"]:
        return True

    col1, col2, col3 = st.columns((1.5, 2.0, 1.5)) 
    
    with col2: 
        st.markdown("""
        <div style="background-color:#F8F9FA; padding:30px; border-radius:15px; border-top: 5px solid #DD4B39; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 50px;">
            <h2 style="color:#1E1E1E; margin-bottom: 5px;">MotorGuard <span style="color:#DD4B39;">AI</span></h2>
            <p style="color:#666666; font-size: 14px; margin-bottom: 20px;">Secure Industrial Portal</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        user = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        if st.button("Login securely", use_container_width=True, type="primary"):
            if user == "admin" and password == "DBA2026":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Incorrect Username or Password")
    return False

def logout():
    st.session_state["authenticated"] = False
    st.rerun()

def create_industrial_gauge(value, title, color_theme, max_val=100):
    safe_value = float(value) # ✅ Fix: Ensure value is a standard Python float
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = safe_value,
        number = {'suffix': "%", 'font': {'size': 24, 'color': color_theme}},
        title = {'text': title, 'font': {'size': 14, 'color': '#555'}},
        gauge = {
            'axis': {'range': (0, max_val), 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color_theme},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': (0, 40), 'color': 'rgba(0, 166, 90, 0.1)'},
                {'range': (40, 75), 'color': 'rgba(243, 156, 18, 0.1)'},
                {'range': (75, 100), 'color': 'rgba(221, 75, 57, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 3},
                'thickness': 0.75,
                'value': safe_value}
        }))
    fig.update_layout(height=200, margin=dict(l=15, r=15, t=35, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# =========================================
# 4. MAIN APP EXECUTION
# =========================================
if not check_password():
    st.stop()

# --- Header ---
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin:0;">⚙️ MotorGuard AI <span style="color:#DD4B39;">Pro</span></h2>
            <p style="margin:0; font-size: 14px; color:#CCCCCC;">Industrial Predictive Maintenance Dashboard V2.0</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.markdown("### 🎛️ SCADA Telemetry")
    motor_type = st.selectbox("Motor Quality Grade", ['L (Low)', 'M (Medium)', 'H (High)'], index=1)
    
    st.markdown("---")
    st.markdown("#### 🌡️ Temperature")
    air_temp = st.number_input("Air Temp [K]", min_value=290.0, max_value=310.0, value=298.0, step=0.1)
    process_temp = st.number_input("Process Temp [K]", min_value=300.0, max_value=320.0, value=308.0, step=0.1)
    
    st.markdown("---")
    st.markdown("#### ⚙️ Mechanical")
    rpm = st.slider("Rotational Speed [RPM]", min_value=1100, max_value=2900, value=1500, step=10)
    torque = st.slider("Torque [Nm]", min_value=10.0, max_value=80.0, value=40.0, step=0.5)
    tool_wear = st.slider("Tool Wear [min]", min_value=0, max_value=260, value=50, step=1)
    
    st.markdown("---")
    analyze_btn = st.button("🔍 Run Full Diagnostics", use_container_width=True, type="primary")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🚪 Secure Logout", use_container_width=True):
        logout()

# --- Analysis & Dashboard ---
if analyze_btn and scaler and model:
    type_dict = {'L (Low)': 0, 'M (Medium)': 1, 'H (High)': 2}
    raw_df = pd.DataFrame({
        'Type': [type_dict[motor_type]],
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rpm],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear]
    })
    
    cols_to_scale = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    processed_df = raw_df.copy()
    processed_df[cols_to_scale] = scaler.transform(raw_df[cols_to_scale])
    
    with st.spinner('MotorGuard AI Engine is analyzing patterns...'):
        prediction_raw = model.predict(processed_df)
        diag_label = str(prediction_raw)
        
        probs = model.predict_proba(processed_df)
        
        p_normal = float(probs * 100) if len(probs) > 0 else 0.0 # ✅ Fix: Enforce float
        p_hdf = float(probs * 100) if len(probs) > 1 else 0.0 # ✅ Fix: Enforce float
        p_osf = float(probs * 100) if len(probs) > 2 else 0.0 # ✅ Fix: Enforce float
        p_pwf = float(probs * 100) if len(probs) > 3 else 0.0 # ✅ Fix: Enforce float
        p_twf = float(probs * 100) if len(probs) > 4 else 0.0 # ✅ Fix: Enforce float
        
        overall_risk_pct = float(100.0 - p_normal) # ✅ Fix: Enforce float

    st.markdown("### 🖥️ AI Diagnostic Summary")
    if diag_label == 'Normal':
        st.markdown(f"""
        <div class="status-safe">
            <h2 style="color:#00A65A; margin:0;">✅ SYSTEM STATUS: {diag_label.upper()}</h2>
            <p style="color:#555; font-size:16px; margin:0;">All sensor readings are within stable operational parameters.</p>
        </div>
        """, unsafe_allow_html=True)
        sys_status = "Safe"
    else:
        st.markdown(f"""
        <div class="status-critical">
            <h2 style="color:#DD4B39; margin:0;">🚨 ALERT DETECTED: {diag_label.upper()}</h2>
            <p style="color:#555; font-size:16px; margin:0;">Immediate intervention required. Risk of failure is high.</p>
        </div>
        """, unsafe_allow_html=True)
        sys_status = "Critical"
        
    st.markdown("---")

    st.markdown("### 📊 Failure Probability Analysis")
    st.markdown("*(Real-time breakdown of potential failure modes based on current telemetry)*")
    
    g1, g2, g3, g4, g5 = st.columns(5)
    
    with g1:
        st.plotly_chart(create_industrial_gauge(overall_risk_pct, "OVERALL RISK", "#1E1E1E"), use_container_width=True)
    with g2:
        st.plotly_chart(create_industrial_gauge(p_hdf, "Heat (HDF)", "#E67E22"), use_container_width=True)
    with g3:
        st.plotly_chart(create_industrial_gauge(p_osf, "Overstrain (OSF)", "#9B59B6"), use_container_width=True)
    with g4:
        st.plotly_chart(create_industrial_gauge(p_pwf, "Power (PWF)", "#E74C3C"), use_container_width=True)
    with g5:
        st.plotly_chart(create_industrial_gauge(p_twf, "Wear (TWF)", "#34495E"), use_container_width=True)

    new_log_raw = pd.DataFrame([{
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Motor Type': motor_type.split(' '),
        'Speed (RPM)': rpm,
        'Overall Risk': f"{overall_risk_pct:.1f}%",
        'Diagnosis': diag_label,
        'Status': sys_status
    }])
    st.session_state['inspection_log'] = pd.concat([new_log_raw, st.session_state['inspection_log']], ignore_index=True)

# =========================================
# 5. INSPECTION HISTORY
# =========================================
st.markdown("---")
st.markdown("### 📋 Automated Inspection Log")
if not st.session_state['inspection_log'].empty:
    st.dataframe(st.session_state['inspection_log'], use_container_width=True, hide_index=True)
else:
    st.info("No active inspection history available in the current session. Run a diagnostic to see logs.")

# =========================================
# 6. FOOTER
# =========================================
st.markdown("""
<br><hr>
<div style='text-align: center; color: #888888; font-size: 12px; font-family: sans-serif;'>
    <strong>MotorGuard AI Pro</strong> | Industrial Diagnostics Engine V2.0<br>
    Developed by <strong>Eng. Mahmoud</strong> © 2026. All Rights Reserved.
</div>
""", unsafe_allow_html=True)