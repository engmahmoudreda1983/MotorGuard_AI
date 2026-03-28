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
    try:
        safe_value = float(value)
    except:
        safe_value = 0.0
        
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
                {'range': (0, 40), 'color': '#E8F5E9'},  
                {'range': (40, 75), 'color': '#FEF9E7'},  
                {'range': (75, 100), 'color': '#FDEDEC'}  
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
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
        'Type': [type_dict.get(motor_type, 1)],
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
        
        prediction_matrix = getattr(model, 'predict')(processed_df)
        diag_label = str(np.array(prediction_matrix).flatten().item(0)).strip()
        
        probs_matrix = getattr(model, 'predict_proba')(processed_df)
        probs_list = np.array(probs_matrix).flatten().tolist()
        
        if hasattr(model, 'classes_'):
            classes = [str(c).strip().upper() for c in getattr(model, 'classes_')]
            probs_dict = dict(zip(classes, probs_list))
            
            p_normal = float(probs_dict.get('NORMAL', 0.0)) * 100.0
            
            p_hdf_raw = float(probs_dict.get('HEAT DISSIPATION FAILURE (HDF)', probs_dict.get('HDF', 0.0))) * 100.0
            p_osf_raw = float(probs_dict.get('OVERSTRAIN FAILURE (OSF)', probs_dict.get('OSF', 0.0))) * 100.0
            p_pwf_raw = float(probs_dict.get('POWER FAILURE (PWF)', probs_dict.get('PWF', 0.0))) * 100.0
            p_twf_raw = float(probs_dict.get('TOOL WEAR FAILURE (TWF)', probs_dict.get('TWF', 0.0))) * 100.0
            
            # الخطر الكلي
            overall_risk_pct = 100.0 - p_normal

            # ✅ التكبير الذكي (Smart Normalization):
            # بنشوف مين أعلى عطل الموديل توقعه، وبنكبره عشان يساوي الخطر الكلي، وباقي الأعطال بتكبر معاه.
            max_fault = max(p_hdf_raw, p_osf_raw, p_pwf_raw, p_twf_raw)
            if max_fault > 0:
                scale_factor = overall_risk_pct / max_fault
                p_hdf = min(100.0, p_hdf_raw * scale_factor)
                p_osf = min(100.0, p_osf_raw * scale_factor)
                p_pwf = min(100.0, p_pwf_raw * scale_factor)
                p_twf = min(100.0, p_twf_raw * scale_factor)
            else:
                p_hdf = p_osf = p_pwf = p_twf = 0.0
        else:
            p_normal = float(probs_list.pop(0)) * 100.0 if len(probs_list) > 0 else 0.0
            p_hdf = p_osf = p_pwf = p_twf = 0.0
            overall_risk_pct = 100.0 - p_normal

    st.markdown("### 🖥️ AI Diagnostic Summary")
    
    if overall_risk_pct < 40.0:
        status_color = "#00A65A" 
        status_bg = "#E8F5E9"
        status_title = "✅ SYSTEM STATUS: NORMAL"
        sys_status = "Safe"
    elif overall_risk_pct < 75.0:
        status_color = "#F39C12" 
        status_bg = "#FEF9E7"
        status_title = f"⚠️ WARNING: ELEVATED RISK ({overall_risk_pct:.1f}%)"
        sys_status = "Warning"
    else:
        status_color = "#DD4B39" 
        status_bg = "#FDEDEC"
        display_label = diag_label if diag_label.upper() != 'NORMAL' else 'FAILURE PREDICTED'
        status_title = f"🚨 DANGER DETECTED: {display_label}"
        sys_status = "Danger"
        
    st.markdown(f"""
    <div style="background-color:{status_bg}; border: 2px solid {status_color}; padding: 15px; border-radius: 10px; text-align: center;">
        <h2 style="color:{status_color}; margin:0;">{status_title}</h2>
        <p style="color:#555; font-size:16px; margin:0;">Real-time SCADA Assessment</p>
    </div>
    """, unsafe_allow_html=True)
        
    st.markdown("---")

    st.markdown("### 📊 Failure Probability Analysis")
    st.markdown("*(Real-time breakdown of potential failure modes scaled to relative risk)*")
    
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
        
    st.markdown("---")
    st.markdown("### 📈 Live Feature Impact Analysis")
    st.markdown("*(Shows the probability distribution of different failure types)*")
    
    impact_data = pd.DataFrame({
        'Failure Mode': ['Heat (HDF)', 'Overstrain (OSF)', 'Power (PWF)', 'Tool Wear (TWF)'],
        'Relative Probability (%)': [p_hdf, p_osf, p_pwf, p_twf]
    }).sort_values(by='Relative Probability (%)', ascending=True)

    fig_bar = px.bar(impact_data, x='Relative Probability (%)', y='Failure Mode', orientation='h', color_discrete_sequence=['#3498DB'])
    fig_bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_bar, use_container_width=True)

    motor_short_name = str(motor_type).split(' ').pop(0)

    new_log_raw = pd.DataFrame([{
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Motor Type': motor_short_name,
        'Speed (RPM)': rpm,
        'Overall Risk': f"{overall_risk_pct:.1f}%",
        'Diagnosis': diag_label,
        'Status': sys_status
    }])
    st.session_state.inspection_log = pd.concat([new_log_raw, st.session_state.inspection_log], ignore_index=True)

# =========================================
# 5. INSPECTION HISTORY
# =========================================
st.markdown("---")
st.markdown("### 📋 Automated Inspection Log")
if not st.session_state.inspection_log.empty:
    st.dataframe(st.session_state.inspection_log, use_container_width=True, hide_index=True)
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