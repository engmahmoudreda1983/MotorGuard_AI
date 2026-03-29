# ==============================================================================
# © 2026 MotorGuard AI | Predictive Maintenance & Fault Diagnostics
# Developed by: Eng. Mahmoud Reda Ibrahim
# Confidential and Proprietary.
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
    .metric-card {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3498DB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

if 'inspection_log' not in st.session_state:
    st.session_state['inspection_log'] = pd.DataFrame(columns=['Timestamp', 'Motor Type', 'Speed (RPM)', 'Risk Level', 'FMEA Diagnosis', 'Status'])

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
            <p style="color:#666666; font-size: 14px; margin-bottom: 20px;">Secure Industrial Diagnostic Portal</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        user = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        if st.button("Secure Login", use_container_width=True, type="primary"):
            if user == "admin" and password == "DBA2026":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Incorrect Username or Password")
                
        st.markdown("""
        <hr><div style='text-align: center; color: #888888; font-size: 11px;'>
            <strong>© 2026 MotorGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
            Confidential and Proprietary. The predictive algorithms and expert system overrides within this application are protected intellectual property.
        </div>
        """, unsafe_allow_html=True)
    return False

def logout():
    st.session_state["authenticated"] = False
    st.rerun()

def create_industrial_gauge(value, title, color_theme, max_val=100, height=200):
    try:
        safe_value = float(value)
    except:
        safe_value = 0.0
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = safe_value,
        number = {'suffix': "%", 'font': {'size': 20 if height<250 else 28, 'color': color_theme}},
        title = {'text': title, 'font': {'size': 13 if height<250 else 18, 'color': '#555'}},
        gauge = {
            'axis': {'range': (0, max_val), 'tickwidth': 1, 'tickcolor': "darkblue", 'tickvals':},
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
    fig.update_layout(height=height, margin=dict(l=15, r=15, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
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

# --- Visual Legend (Risk Zones) ---
st.markdown("""
<div style="display:flex; justify-content:space-between; margin-bottom:15px; font-size: 12px; font-weight:bold;">
    <div style="width: 39%; background-color:#E8F5E9; color:#00A65A; text-align:center; padding:5px; border-radius:5px; border:1px solid #00A65A;">✅ SAFE: 0% - 40%</div>
    <div style="width: 34%; background-color:#FEF9E7; color:#F39C12; text-align:center; padding:5px; border-radius:5px; border:1px solid #F39C12;">⚠️ WARNING: 40% - 75%</div>
    <div style="width: 25%; background-color:#FDEDEC; color:#DD4B39; text-align:center; padding:5px; border-radius:5px; border:1px solid #DD4B39;">🚨 DANGER: 75% - 100%</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.markdown("### 🎛️ Live Sensor Telemetry")
    motor_options = {
        'L (Low - Standard Build)': 0,
        'M (Medium - Commercial)': 1,
        'H (High - Industrial Grade)': 2
    }
    motor_type = st.selectbox(
        "Motor Quality Grade", 
        list(motor_options.keys()), 
        index=1,
        help="L: Low quality (More prone to failure). M: Medium quality. H: High quality (Durable industrial grade)."
    )
    
    st.markdown("---")
    st.markdown("#### 🌡️ Thermal Metrics")
    air_temp = st.number_input("Ambient Air Temp [K]", min_value=290.0, max_value=310.0, value=298.0, step=0.1, help="Temperature of the surrounding environment in Kelvin. Range: 290K - 310K.")
    process_temp = st.number_input("Motor Core Temp [K]", min_value=300.0, max_value=320.0, value=308.0, step=0.1, help="Operating temperature of the motor core. Range: 300K - 320K. (Needs to be higher than Ambient Temp for heat dissipation).")
    
    st.markdown("---")
    st.markdown("#### ⚙️ Mechanical Metrics")
    rpm = st.slider("Rotational Speed [RPM]", min_value=1100, max_value=2900, value=1500, step=10, help="Motor shaft speed. Extreme high speeds increase Power Failure (PWF) risk.")
    torque = st.slider("Applied Torque [Nm]", min_value=10.0, max_value=80.0, value=40.0, step=0.5, help="Twisting force applied by the motor. High torque increases Overstrain Failure (OSF) risk.")
    tool_wear = st.slider("Tool Wear Level [min]", min_value=0, max_value=260, value=50, step=1, help="Usage duration of the attached tool. Exceeding 200 mins drastically increases Tool Wear Failure (TWF) risk.")
    
    st.markdown("---")
    analyze_btn = st.button("🔍 Run Full Diagnostics", use_container_width=True, type="primary")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🚪 Secure Logout", use_container_width=True):
        logout()

# --- Analysis & Dashboard ---
if analyze_btn and scaler and model:
    raw_df = pd.DataFrame({
        'Type': [motor_options[motor_type]],
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rpm],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear]
    })
    
    cols_to_scale = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    processed_df = raw_df.copy()
    processed_df[cols_to_scale] = scaler.transform(raw_df[cols_to_scale])
    
    with st.spinner('AI Engine & Expert Systems are analyzing patterns...'):
        
        # 1. AI PREDICTION
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
            
            raw_overall_risk = 100.0 - p_normal
        else:
            p_normal = float(probs_list.pop(0)) * 100.0 if len(probs_list) > 0 else 0.0
            raw_overall_risk = 100.0 - p_normal
            p_hdf_raw = p_osf_raw = p_pwf_raw = p_twf_raw = 0.0

        # 2. EXPERT SYSTEM OVERRIDES (Physical fail-safes)
        # Force high risk if physical constraints are clearly violated
        if tool_wear >= 240:
            raw_overall_risk = max(raw_overall_risk, 95.0)
            p_twf_raw = max(p_twf_raw, 95.0)
        
        if (torque * rpm) > 90000:
            raw_overall_risk = max(raw_overall_risk, 96.0)
            p_pwf_raw = max(p_pwf_raw, 96.0)
            
        if (torque * tool_wear) > 12000:
            raw_overall_risk = max(raw_overall_risk, 94.0)
            p_osf_raw = max(p_osf_raw, 94.0)
            
        if (process_temp - air_temp) <= 5 and process_temp > 310:
            raw_overall_risk = max(raw_overall_risk, 92.0)
            p_hdf_raw = max(p_hdf_raw, 92.0)

        # 3. SMART NORMALIZATION
        if raw_overall_risk >= 80.0:
            overall_risk_pct = 100.0
        else:
            overall_risk_pct = raw_overall_risk

        max_fault = max(p_hdf_raw, p_osf_raw, p_pwf_raw, p_twf_raw)
        if max_fault > 0:
            scale_factor = overall_risk_pct / max_fault
            p_hdf = min(100.0, p_hdf_raw * scale_factor)
            p_osf = min(100.0, p_osf_raw * scale_factor)
            p_pwf = min(100.0, p_pwf_raw * scale_factor)
            p_twf = min(100.0, p_twf_raw * scale_factor)
        else:
            p_hdf = p_osf = p_pwf = p_twf = 0.0

        ai_confidence = max(p_normal, max_fault) if raw_overall_risk < 80 else 99.9

    # --- TOP ROW: System Status & Overall Risk ---
    if overall_risk_pct < 40.0:
        status_color = "#00A65A" 
        status_bg = "#E8F5E9"
        status_title = "✅ SYSTEM NORMAL"
        sys_status = "Safe"
    elif overall_risk_pct < 75.0:
        status_color = "#F39C12" 
        status_bg = "#FEF9E7"
        status_title = f"⚠️ ELEVATED RISK"
        sys_status = "Warning"
    else:
        status_color = "#DD4B39" 
        status_bg = "#FDEDEC"
        display_label = diag_label if diag_label.upper() != 'NORMAL' else 'FAILURE PREDICTED'
        status_title = f"🚨 DANGER: {display_label}"
        sys_status = "Danger"
        
    st.markdown(f"""
    <div style="background-color:{status_bg}; border: 2px solid {status_color}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom:20px;">
        <h2 style="color:{status_color}; margin:0;">{status_title}</h2>
        <p style="color:#555; font-size:16px; margin:0;">Live Assessment based on AI & Expert Overrides</p>
    </div>
    """, unsafe_allow_html=True)
        
    # --- MIDDLE ROW: Key Metrics & Master Gauge ---
    colA, colB = st.columns([1, 1.5])
    
    with colA:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color:#666; margin-bottom:5px;">AI Confidence Score</h4>
            <h2 style="color:#3498DB; margin:0;">{ai_confidence:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        highest_fault_name = "None"
        if overall_risk_pct >= 40:
            fault_dict = {'Heat (HDF)': p_hdf, 'Overstrain (OSF)': p_osf, 'Power (PWF)': p_pwf, 'Wear (TWF)': p_twf}
            highest_fault_name = max(fault_dict, key=fault_dict.get)
            
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {status_color};">
            <h4 style="color:#666; margin-bottom:5px;">Primary FMEA Concern</h4>
            <h2 style="color:{status_color}; margin:0;">{highest_fault_name}</h2>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.plotly_chart(create_industrial_gauge(overall_risk_pct, "MASTER OVERALL RISK", status_color, height=280), use_container_width=True)

    st.markdown("---")
    
    # --- BOTTOM ROW: The 4 Fault Sub-Gauges ---
    st.markdown("### 📊 FMEA Fault Probability Analysis")
    st.markdown("*(Root cause distribution derived from Random Forest arrays)*")
    
    g1, g2, g3, g4 = st.columns(4)
    
    with g1:
        st.plotly_chart(create_industrial_gauge(p_hdf, "Heat (HDF)", "#E67E22", height=180), use_container_width=True)
    with g2:
        st.plotly_chart(create_industrial_gauge(p_osf, "Overstrain (OSF)", "#9B59B6", height=180), use_container_width=True)
    with g3:
        st.plotly_chart(create_industrial_gauge(p_pwf, "Power (PWF)", "#E74C3C", height=180), use_container_width=True)
    with g4:
        st.plotly_chart(create_industrial_gauge(p_twf, "Wear (TWF)", "#34495E", height=180), use_container_width=True)

    motor_short_name = str(motor_type).split(' ')

    new_log_raw = pd.DataFrame([{
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Motor Type': motor_short_name,
        'Speed (RPM)': rpm,
        'Risk Level': f"{overall_risk_pct:.1f}%",
        'FMEA Diagnosis': diag_label if overall_risk_pct > 40 else "NORMAL",
        'Status': sys_status
    }])
    st.session_state.inspection_log = pd.concat([new_log_raw, st.session_state.inspection_log], ignore_index=True)

# =========================================
# 5. INSPECTION HISTORY (FMEA Log)
# =========================================
st.markdown("---")
st.markdown("### 📋 FMEA Event Log")
if not st.session_state.inspection_log.empty:
    st.dataframe(st.session_state.inspection_log, use_container_width=True, hide_index=True)
else:
    st.info("No active events logged in the current session. Run a diagnostic to populate the FMEA table.")

# =========================================
# 6. FOOTER
# =========================================
st.markdown("""
<br><hr>
<div style='text-align: center; color: #888888; font-size: 11px; font-family: sans-serif; line-height: 1.5;'>
    <strong>© 2026 MotorGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
    Confidential and Proprietary. The predictive algorithms and expert system overrides within this application are protected intellectual property.
</div>
""", unsafe_allow_html=True)