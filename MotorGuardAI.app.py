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
# 1. PAGE SETUP & THEME
# =========================================
st.set_page_config(page_title="MotorGuard AI Pro", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

# --- Initialize Session State for Inspection Log & Authentication ---
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

# --- Hardcoded Feature Importances for Analysis Plot ---
data_pattern_importances = pd.DataFrame({
    'Feature': ['Torque', 'RPM', 'Wear', 'AirTemp', 'ProcTemp', 'Type'],
    'Impact': [0.3111, 0.2973, 0.2063, 0.1020, 0.0658, 0.0174]
})
data_pattern_importances = data_pattern_importances.sort_values(by='Impact', ascending=True)

# =========================================
# 3. HELPER FUNCTIONS
# =========================================

# --- Authentication Logic ---
def check_password():
    def password_entered():
        if st.session_state["user"] == "user" and st.session_state["pass"] == "pass":
            st.session_state["authenticated"] = True
            del st.session_state["user"]
            del st.session_state["pass"]
        else:
            st.session_state["authenticated"] = False
            st.error("😕 Username or password incorrect.")

    if not st.session_state["authenticated"]:
        # ✅ التصحيح الأول هنا: إضافة نسب التقسيم داخل الأقواس
        cols = st.columns() 
        with cols:
            st.markdown("""
            <div style="background-color:#F5F5F5; padding:30px; border-radius:15px; border-bottom: 5px solid #DD4B39; text-align: center;">
                <h1 style="color:#1E1E1E; margin:0; font-size: 50px;">MotorGuard <span style="color:#DD4B39;">AI</span></h1>
                <p style="color:#888888; margin:0;">Login to Secure Industrial Dashboard</p>
            </div>
            <br>
            """, unsafe_allow_html=True)
            st.text_input("Username", key="user")
            st.text_input("Password", type="password", key="pass")
            st.button("Login Securely", on_click=password_entered, use_container_width=True, type="primary")
            return False
    else:
        return True

# --- Gauge Chart Creation ---
def create_industrial_gauge(value, title, unit, max_val, thresholds):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': f"{title} ({unit})", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [None, max_val]},
            'bar': {'color': "#1E1E1E"},
            'steps': [
                {'range': [0, thresholds['warn']], 'color': "#00A65A"},
                {'range': [thresholds['warn'], thresholds['danger']], 'color': "#f39c12"},
                {'range': [thresholds['danger'], max_val], 'color': "#DD4B39"}
            ],
            'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': thresholds['critical']}
        }))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig

# =========================================
# 4. MAIN APP EXECUTION
# =========================================

if not check_password():
    st.stop()

# --- Header & Copyrights ---
st.markdown("""
<div style="background-color:#1E1E1E; padding:15px; border-radius:10px; border-left: 8px solid #DD4B39; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <h1 style="color:white; margin:0;">⚙️ MotorGuard AI <span style="color:#DD4B39;">Pro</span></h1>
        <p style="color:#B0B0B0; margin:0; font-size: 14px;">Next-Generation Predictive Maintenance System (V1.0)</p>
    </div>
    <div style="text-align: right; color:#888888; font-size: 12px;">
        © 2026 MotorGuard AI. All Rights Reserved.<br>
        Developed for Industrial Assets Monitoring
    </div>
</div>
<br>
""", unsafe_allow_html=True)

# =========================================
# 5. INDUSTRIAL SCADA SIDEBAR
# =========================================
with st.sidebar:
    st.markdown("## 🎛️ SCADA Inputs")
    st.markdown("Enter Live Motor Telemetry:")
    
    with st.expander("🛠️ Motor Quality"):
        motor_type = st.selectbox("Motor Type (Quality Level)", ['L (Low)', 'M (Medium)', 'H (High)'], index=1)
        st.markdown("*Quality ranking in terms of material strength.*")
    
    st.markdown("---")
    st.subheader("🌡️ Thermal Sensors")
    air_temp = st.number_input("Air Temperature [K]", min_value=290.0, max_value=310.0, value=298.0, step=0.1, help="Temperature of ambient air near the motor.")
    process_temp = st.number_input("Process Temperature [K]", min_value=300.0, max_value=320.0, value=308.0, step=0.1, help="Temperature of the internal winding process.")
    
    st.markdown("---")
    st.subheader("⚙️ Mechanical Sensors")
    rpm = st.slider("Rotational Speed [rpm]", min_value=1100, max_value=2900, value=1500, step=10, help="Motor output speed.")
    torque = st.slider("Torque [Nm]", min_value=10.0, max_value=80.0, value=40.0, step=0.5, help="Motor output torque/force.")
    tool_wear = st.slider("Tool Wear [min]", min_value=0, max_value=260, value=50, step=1, help="Cumulative wear time on the internal components.")
    
    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze Motor State", use_container_width=True, type="primary")

# =========================================
# 6. DIAGNOSTICS & RESULTS
# =========================================
input_raw_display = None

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
    
    input_raw_display = raw_df.copy()
    cols_to_scale = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    processed_df = raw_df.copy()
    processed_df[cols_to_scale] = scaler.transform(raw_df[cols_to_scale])
    
    with st.spinner('MotorGuard AI is analyzing sensor patterns...'):
        prediction_label = model.predict(processed_df)
        probabilities = model.predict_proba(processed_df)
        
        # حساب نسبة الخطر الإجمالية
        overall_failure_risk_pct = sum(probabilities[1:]) * 100
        overall_failure_risk_pct = min(overall_failure_risk_pct, 100) 

    if prediction_label == 'Normal':
        status_color = "#00A65A"
        status_bg = "#E8F5E9"
        status_title = "SYSTEM SECURE"
        status_msg = "Current diagnosis is Normal Operation. All sensors are within safe patterns."
        sys_status_log = "Safe"
    else:
        status_color = "#DD4B39"
        status_bg = "#FDEDEC"
        status_title = "ALERT: INTERVENTION REQUIRED"
        status_msg = f"Potential imminent failure detected: **{prediction_label.upper()}**.\nInspect the asset immediately."
        sys_status_log = "Critical"

    st.markdown(f"""
    <div style="background-color:{status_bg}; padding:20px; border-radius:10px; border: 2px solid {status_color};">
        <h3 style="color:{status_color}; margin:0;">{status_title}</h3>
        <p style="font-size: 16px; margin-top: 5px;">{status_msg}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Live Telemetry & Advanced Analysis")
    
    # ✅ التصحيح الثاني هنا: إضافة نسب التقسيم داخل الأقواس
    g_cols = st.columns() 
    
    with g_cols:
        st.help(help="Cumulative wear time on the internal components (min). Failure expected after ~200 mins.")
        st.plotly_chart(create_industrial_gauge(tool_wear, "Tool Wear", "min", 260, {'warn': 150, 'danger': 200, 'critical': 240}), use_container_width=True)
    with g_cols:
        st.help(help="Temperature of ambient air near the motor [K]. Danger above ~302K.")
        st.plotly_chart(create_industrial_gauge(air_temp, "Air Temp", "K", 310, {'warn': 301, 'danger': 303, 'critical': 305}), use_container_width=True)
    with g_cols:
        st.help(help="Temperature of the internal winding process [K]. Heat failure occurs if Process - Air is high.")
        st.plotly_chart(create_industrial_gauge(process_temp, "Process Temp", "K", 320, {'warn': 310, 'danger': 315, 'critical': 318}), use_container_width=True)
    with g_cols:
        st.help(help="Aggregated Probability of Any Failure. Sum of probabilities for HDF, OSF, PWF, TWF, RNF.")
        st.plotly_chart(create_industrial_gauge(overall_failure_risk_pct, "OVERALL FAILURE RISK", "%", 100, {'warn': 25, 'danger': 50, 'critical': 80}), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 AI Root Cause Analysis (Root Cause Determination)")
    st.help(help="Dynamic Bar Chart showing the factors contributing most to the current failure diagnosis.")

    fig_pattern = px.bar(data_pattern_importances, x='Impact', y='Feature', orientation='h',
                     color_discrete_sequence=['#DD4B39'])
    fig_pattern.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_pattern, use_container_width=True)
    
    if prediction_label != 'Normal':
        st.info(f"Root Cause of {prediction_label}: Refer to the top bars of the 'AI Pattern Analysis' Chart.")

    new_log_raw = pd.DataFrame([{
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Type': motor_type.split(' '),
        'Overall Risk': f"{overall_failure_risk_pct:.1f}%",
        'RPM': rpm,
        'Diagnosis': prediction_label,
        'Status': sys_status_log
    }])
    st.session_state['inspection_log'] = pd.concat([new_log_raw, st.session_state['inspection_log']], ignore_index=True)

    report_text = f"""
=========================================
 MotorGuard AI - Inspection Report © 2026
 Industrial Predictive Diagnostics Engine V1.0
=========================================
Date/Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Generated by Eng. Mahmoud System

--- Motor Identification ---
Motor Type (Quality Level): {motor_type}

--- SCADA Sensor Telemetry (Raw) ---
Air Temperature: {air_temp} K
Process Temperature: {process_temp} K
Rotational Speed: {rpm} RPM
Torque: {torque} Nm
Tool Wear: {tool_wear} min

--- AI Multi-Label Diagnosis ---
Overall Failure Risk: {overall_failure_risk_pct:.1f}%
System Security Status: {sys_status_log.upper()}
AI Predicted Failure Mode: **{prediction_label.upper()}**
=========================================
    """
    st.download_button(
        label="📥 Download Secure Diagnostic Report (TXT)",
        data=report_text,
        file_name=f"MotorGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

# =========================================
# 7. INSPECTION HISTORY LOG
# =========================================
st.markdown("---")
st.markdown("### 📋 Automated Inspection Log (Session History)")
if not st.session_state['inspection_log'].empty:
    st.dataframe(st.session_state['inspection_log'], use_container_width=True)
else:
    st.info("No active inspection history available in current session.")

# =========================================
# 8. INDUSTRIAL FOOTER & RIGHTS
# =========================================
st.markdown("""
<hr>
<div style='text-align: center; color: #888888; font-size: 11px; font-family: sans-serif; padding-top: 10px;'>
    <p style='margin: 0; padding-bottom: 5px;'>⚙️ <strong>MotorGuard AI Pro</strong> | Industrial Diagnostics V1.0</p>
    <p style='margin: 0;'>Developed by <strong>Eng. Mahmoud</strong> © 2026. All Rights Reserved. This system is for secure industrial monitoring.</p>
</div>
""", unsafe_allow_html=True)