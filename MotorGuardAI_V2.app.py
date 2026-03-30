# ==============================================================================
# © 2026 PowerGuard AI | Enterprise Predictive Maintenance & FMEA
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
st.set_page_config(page_title="PowerGuard AI Pro", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

def highlight_fmea(row):
    status = row.get('Status', '')
    if status == 'Safe': return ['background-color: #E8F5E9; color: #00A65A; font-weight: bold'] * len(row)
    if status == 'Warning': return ['background-color: #FEF9E7; color: #D35400; font-weight: bold'] * len(row)
    if status == 'Danger': return ['background-color: #FDEDEC; color: #C0392B; font-weight: bold'] * len(row)
    return [''] * len(row)

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #DD4B39;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
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
    .footer-text { text-align: center; color: #888; font-size: 11px; margin-top: 30px; line-height: 1.5; }
    </style>
""", unsafe_allow_html=True)

# --- Session States ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'motor_risk' not in st.session_state: st.session_state.motor_risk = 0.0
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF': 0.0, 'OSF': 0.0, 'PWF': 0.0, 'TWF': 0.0}
if 'inspection_log' not in st.session_state:
    cols = ['Timestamp', 'Asset', 'Risk Level', 'Primary FMEA', 'Status', 'Grade', 'Air Temp [K]', 'Core Temp [K]', 'RPM', 'Torque [Nm]', 'Wear [min]', 'Vib RMS', 'Vib Kurt']
    st.session_state['inspection_log'] = pd.DataFrame(columns=cols)

# =========================================
# 2. AI MODELS LOADING
# =========================================
@st.cache_resource
def load_ai_assets():
    try:
        s1 = joblib.load('scaler.pkl')
        m1 = joblib.load('motorguard_model.pkl')
        s2 = joblib.load('nasa_bearing_scaler.pkl')
        m2 = joblib.load('nasa_bearing_model.pkl')
        return s1, m1, s2, m2, True
    except:
        return None, None, None, None, False

scaler1, model1, scaler2, model2, models_ready = load_ai_assets()

# =========================================
# 3. AUTHENTICATION (AS PER IMAGE)
# =========================================
def check_password():
    if st.session_state.authenticated: return True
    col1, col2, col3 = st.columns([1.5, 2.0, 1.5]) 
    with col2: 
        st.markdown("""
        <div style="background-color:#F8F9FA; padding:30px; border-radius:15px; border-top: 5px solid #DD4B39; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 50px;">
            <h2 style="color:#1E1E1E; margin-bottom: 5px;">PowerGuard <span style="color:#DD4B39;">AI</span></h2>
            <p style="color:#666666; font-size: 14px; margin-bottom: 20px;">Secure Industrial Diagnostic Portal</p>
        </div><br>
        """, unsafe_allow_html=True)
        user = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        if st.button("Secure Login", use_container_width=True, type="primary"):
            if user == "admin" and password == "DBA2026":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect Username or Password")
        st.markdown("""
        <div class="footer-text">
            <strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
            Confidential and Proprietary. The predictive algorithms and expert system overrides within this application are protected intellectual property.
        </div>
        """, unsafe_allow_html=True)
    return False

def create_gauge(value, title, color, h=250):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value),
        number={'suffix': "%", 'font': {'size': 26, 'color': color}},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range':},
            'bar': {'color': color},
            'steps': [{'range':, 'color': '#E8F5E9'}, {'range':, 'color': '#FEF9E7'}, {'range':, 'color': '#FDEDEC'}],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': float(value)}
        }))
    fig.update_layout(height=h, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

if not check_password(): st.stop()

# =========================================
# 4. MAIN APP
# =========================================
with st.sidebar:
    st.markdown("### 🖥️ Edge Control")
    if st.button("🚪 Secure Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    st.markdown("---")
    if models_ready: st.success("✅ Edge Sensors Connected.")
    else: st.error("⚠️ AI Assets Missing.")

st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin:0;">⚙️ PowerGuard AI <span style="color:#DD4B39;">Enterprise</span></h2>
            <p style="margin:0; font-size: 14px; color:#CCCCCC;">Advanced Asset Performance Management System</p>
        </div>
        <div style="text-align: right;"><h4 style="margin:0; color:#00ff88;">⚡ LIVE TELEMETRY</h4></div>
    </div>
</div>
""", unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs(["⚙️ Motor Diagnostics", "📳 Bearing Vibration", "🔗 Reliability Fusion", "📊 Executive Summary"])

# --- TAB 1: MOTOR ---
with t1:
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.markdown("#### 🎛️ Live Sensor Telemetry")
        m_opt = {'L (Low)': 0, 'M (Medium)': 1, 'H (High)': 2}
        m_type = st.selectbox("Motor Quality Grade ❓", list(m_opt.keys()), index=1)
        air_t = st.number_input("Ambient Air Temp [K] ❓", 290.0, 310.0, 298.0, 0.1)
        proc_t = st.number_input("Motor Core Temp [K] ❓", 300.0, 320.0, 308.0, 0.1)
        rpm = st.slider("Rotational Speed [RPM] ❓", 1100, 2900, 1500, 10)
        trq = st.slider("Applied Torque [Nm] ❓", 10.0, 80.0, 40.0, 0.5)
        wear = st.slider("Tool Wear [min] ❓", 0, 260, 50, 1)

    with c2:
        if models_ready:
            # Original V1/V2 Data Processing
            raw = pd.DataFrame({'Type': [m_opt[m_type]], 'Air temperature [K]': [air_t], 'Process temperature [K]': [proc_t], 'Rotational speed [rpm]': [rpm], 'Torque [Nm]': [trq], 'Tool wear [min]': [wear]})
            sc_raw = raw.copy()
            sc_raw[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']] = scaler1.transform(raw[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']])
            
            probs = getattr(model1, 'predict_proba')(sc_raw)
            probs_list = np.array(probs).flatten().tolist()
            classes = [str(c).strip().upper() for c in getattr(model1, 'classes_')]
            p_dict = dict(zip(classes, probs_list))
            
            p_norm = p_dict.get('NORMAL', 0.0) * 100.0
            p_hdf_r = p_dict.get('HEAT DISSIPATION FAILURE (HDF)', p_dict.get('HDF', 0.0)) * 100.0
            p_osf_r = p_dict.get('OVERSTRAIN FAILURE (OSF)', p_dict.get('OSF', 0.0)) * 100.0
            p_pwf_r = p_dict.get('POWER FAILURE (PWF)', p_dict.get('PWF', 0.0)) * 100.0
            p_twf_r = p_dict.get('TOOL WEAR FAILURE (TWF)', p_dict.get('TWF', 0.0)) * 100.0
            
            raw_risk = 100.0 - p_norm
            overall_m_risk = 100.0 if raw_risk >= 85.0 else raw_risk
            
            max_f = max(p_hdf_r, p_osf_r, p_pwf_r, p_twf_r)
            if max_f > 0:
                sf = overall_m_risk / max_f
                p_hdf, p_osf, p_pwf, p_twf = p_hdf_r*sf, p_osf_r*sf, p_pwf_r*sf, p_twf_r*sf
            else: p_hdf = p_osf = p_pwf = p_twf = 0.0

            st.session_state.motor_risk = overall_m_risk
            st.session_state.motor_faults = {'HDF': p_hdf, 'OSF': p_osf, 'PWF': p_pwf, 'TWF': p_twf}
            
            s_color = "#00A65A" if overall_m_risk < 40 else ("#F39C12" if overall_m_risk < 75 else "#DD4B39")
            s_bg = "#E8F5E9" if overall_m_risk < 40 else ("#FEF9E7" if overall_m_risk < 75 else "#FDEDEC")
            diag = "NORMAL" if overall_m_risk < 40 else max(st.session_state.motor_faults, key=st.session_state.motor_faults.get)

            cA, cB = st.columns([1, 1.5])
            with cA:
                st.markdown(f"<div style='background-color:{s_bg}; border: 2px solid {s_color}; padding: 15px; border-radius: 8px; text-align: center;'><h3 style='color:{s_color}; margin:0;'>{'SYSTEM NORMAL' if overall_m_risk < 40 else 'RISK DETECTED'}</h3></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><h4>AI Confidence</h4><h2 style='color:#3498DB;'>{p_norm if overall_m_risk < 50 else overall_m_risk:.1f}%</h2></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card' style='border-left-color:{s_color};'><h4>Diagnosis</h4><h2 style='color:{s_color};'>{diag}</h2></div>", unsafe_allow_html=True)
            with cB:
                st.plotly_chart(create_gauge(overall_m_risk, "MASTER OVERALL RISK", s_color, h=280), use_container_width=True)

            st.markdown("---")
            g1, g2, g3, g4 = st.columns(4)
            with g1: st.plotly_chart(create_gauge(p_hdf, "Heat (HDF)", "#E67E22", h=160), use_container_width=True)
            with g2: st.plotly_chart(create_gauge(p_osf, "Strain (OSF)", "#9B59B6", h=160), use_container_width=True)
            with g3: st.plotly_chart(create_gauge(p_pwf, "Power (PWF)", "#E74C3C", h=160), use_container_width=True)
            with g4: st.plotly_chart(create_gauge(p_twf, "Wear (TWF)", "#34495E", h=160), use_container_width=True)

            # Auto-Log
            new_log = pd.DataFrame([{'Timestamp': datetime.now().strftime("%H:%M:%S"), 'Asset': 'Motor Core', 'Risk Level': f"{overall_m_risk:.1f}%", 'Primary FMEA': diag, 'Status': 'Safe' if overall_m_risk < 40 else ('Warning' if overall_m_risk < 75 else 'Danger'), 'Grade': m_type, 'Air Temp [K]': air_t, 'Core Temp [K]': proc_t, 'RPM': rpm, 'Torque [Nm]': trq, 'Wear [min]': wear, 'Vib RMS': '-', 'Vib Kurt': '-'}])
            st.session_state.inspection_log = pd.concat([new_log, st.session_state.inspection_log], ignore_index=True).head(30)

# --- TAB 2: BEARING ---
with t2:
    cv1, cv2 = st.columns([1.5, 1])
    with cv1:
        st.markdown("#### 🎚️ Simulator & Key Metrics")
        sim = st.slider("Degradation Phase (%) ❓", 0, 100, 10)
        r_val, k_val = 0.05 + (sim/100.0)**2 * 0.20, 3.0 + (sim/100.0)**1.5 * 5.0
        st.columns(4).metric("RMS ❓", f"{r_val:.3f}")
    with cv2:
        st.session_state.bearing_risk = 100.0 if sim >= 85.0 else sim
        st.plotly_chart(create_gauge(st.session_state.bearing_risk, "BEARING RISK", "#F39C12", h=280), use_container_width=True)
    
    new_log_b = pd.DataFrame([{'Timestamp': datetime.now().strftime("%H:%M:%S"), 'Asset': 'Bearing', 'Risk Level': f"{st.session_state.bearing_risk:.1f}%", 'Primary FMEA': 'Vibration', 'Status': 'Safe' if st.session_state.bearing_risk < 40 else 'Warning', 'Grade': '-', 'Air Temp [K]': '-', 'Core Temp [K]': '-', 'RPM': '-', 'Torque [Nm]': '-', 'Wear [min]': '-', 'Vib RMS': round(r_val,3), 'Vib Kurt': round(k_val,2)}])
    st.session_state.inspection_log = pd.concat([new_log_b, st.session_state.inspection_log], ignore_index=True).head(30)

# --- TAB 3: FUSION ---
with t3:
    m_f, b_f = st.session_state.motor_risk / 100.0, st.session_state.bearing_risk / 100.0
    int_risk = (1.0 - ((1.0 - m_f) * (1.0 - b_f))) * 100.0
    st.markdown(f"### 🔗 Master Reliability Fusion: {int_risk:.1f}%")
    st.plotly_chart(create_gauge(int_risk, "SYSTEM RISK", "#DD4B39" if int_risk > 75 else "#00ff88", h=300), use_container_width=True)

# --- TAB 4: FMEA LOG ---
with t4:
    st.markdown("### 📊 Comprehensive Diagnostic Log")
    st.dataframe(st.session_state.inspection_log.style.apply(highlight_fmea, axis=1), use_container_width=True, hide_index=True)

# --- FOOTER ---
st.markdown("""
<div class="footer-text">
    <hr>
    <strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
    Confidential and Proprietary. The predictive algorithms and expert system overrides within this application are protected intellectual property.
</div>
""", unsafe_allow_html=True)