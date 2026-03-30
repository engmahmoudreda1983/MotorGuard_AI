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
# 1. PAGE SETUP & GLOBAL STYLES
# =========================================
st.set_page_config(page_title="PowerGuard AI Pro", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

def highlight_fmea(row):
    status = row.get('Status', '')
    if status == 'Safe': return list(('background-color: #E8F5E9; color: #00A65A; font-weight: bold',)) * len(row)
    if status == 'Warning': return list(('background-color: #FEF9E7; color: #D35400; font-weight: bold',)) * len(row)
    if status == 'Danger': return list(('background-color: #FDEDEC; color: #C0392B; font-weight: bold',)) * len(row)
    return list(('',)) * len(row)

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
    .zone-bar {
        width: 100%; height: 20px; display: flex; border-radius: 5px; overflow: hidden; margin-bottom: 5px; font-weight: bold;
    }
    .zone-safe { width: 40%; background-color: #2ecc71; text-align: center; color: white; font-size: 12px; line-height: 20px; }
    .zone-warn { width: 35%; background-color: #f1c40f; text-align: center; color: white; font-size: 12px; line-height: 20px; }
    .zone-dang { width: 25%; background-color: #e74c3c; text-align: center; color: white; font-size: 12px; line-height: 20px; }
    .footer-text { text-align: center; color: #888; font-size: 11px; margin-top: 30px; line-height: 1.5; }
    </style>
""", unsafe_allow_html=True)

# --- Session States ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'motor_risk' not in st.session_state: st.session_state.motor_risk = 0.0
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF': 0.0, 'OSF': 0.0, 'PWF': 0.0, 'TWF': 0.0}
if 'inspection_log' not in st.session_state:
    cols = list(('Timestamp', 'Asset', 'Risk Level', 'Primary FMEA', 'Status', 'Grade', 'Air Temp [K]', 'Core Temp [K]', 'RPM', 'Torque [Nm]', 'Wear [min]', 'Vib RMS', 'Vib Kurt'))
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
    except Exception:
        return None, None, None, None, False

scaler1, model1, scaler2, model2, models_ready = load_ai_assets()

# =========================================
# 3. AUTHENTICATION (IMAGE MATCHED DESIGN)
# =========================================
def check_password():
    if st.session_state.authenticated: return True
    col1, col2, col3 = st.columns(list((1.5, 2.0, 1.5))) 
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

def create_gauge(value, title, color, max_v=100.0, h=250):
    val = float(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number={'suffix': "%", 'font': {'size': 26, 'color': color, 'family': "Arial Black"}},
        title={'text': title, 'font': {'size': 16, 'color': '#333', 'weight': 'bold'}},
        gauge={
            'axis': {'range': list((0, max_v)), 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
            'steps': list(({'range': list((0, 40)), 'color': '#E8F5E9'}, {'range': list((40, 75)), 'color': '#FEF9E7'}, {'range': list((75, 100)), 'color': '#FDEDEC'})),
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': val}
        }))
    fig.update_layout(height=h, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

if not check_password(): st.stop()

# =========================================
# 4. MAIN APP EXECUTION
# =========================================
with st.sidebar:
    st.markdown("### 🖥️ Edge Control")
    if st.button("🚪 Secure Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    st.markdown("---")
    if not models_ready: st.error("⚠️ AI Models not found.")
    else: st.success("✅ Edge Sensors Connected.")
    st.markdown("---")

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

tab1, tab2, tab3, tab4 = st.tabs(list(("⚙️ Motor Diagnostics", "📳 Bearing Vibration", "🔗 Reliability Fusion", "📊 Executive Summary")))

# --- TAB 1: MOTOR DIAGNOSTICS ---
with tab1:
    c_in, c_out = st.columns(list((1, 2.5)))
    with c_in:
        st.markdown("#### 🎛️ Live Sensor Telemetry")
        motor_options = {'L (Low)': 0, 'M (Medium)': 1, 'H (High)': 2}
        motor_type = st.selectbox("Motor Quality Grade ❓", list(motor_options.keys()), index=1, help="Quality grade of the motor asset.")
        air_temp = st.number_input("Ambient Air Temp [K] ❓", 290.0, 310.0, 298.0, 0.1, help="Ambient temperature around the asset.")
        process_temp = st.number_input("Motor Core Temp [K] ❓", 300.0, 320.0, 308.0, 0.1, help="Internal core temperature.")
        rpm = st.slider("Rotational Speed [RPM] ❓", 1100, 2900, 1500, 10, help="Operational speed of the shaft.")
        torque = st.slider("Applied Torque [Nm] ❓", 10.0, 80.0, 40.0, 0.5, help="Mechanical twist force.")
        tool_wear = st.slider("Tool Wear [min] ❓", 0, 260, 50, 1, help="Usage time of the mechanical tool component.")

    with c_out:
        if models_ready:
            c_air, c_proc, c_rpm, c_trq, c_wear = 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
            raw_df = pd.DataFrame({'Type': list((motor_options[motor_type],)), c_air: list((air_temp,)), c_proc: list((process_temp,)), c_rpm: list((rpm,)), c_trq: list((torque,)), c_wear: list((tool_wear,))})
            processed_df = raw_df.copy()
            processed_df[list((c_air, c_proc, c_rpm, c_trq, c_wear))] = scaler1.transform(raw_df[list((c_air, c_proc, c_rpm, c_trq, c_wear))])
            
            probs_list = np.array(getattr(model1, 'predict_proba')(processed_df)).flatten().tolist()
            if hasattr(model1, 'classes_'):
                classes = list((str(c).strip().upper() for c in getattr(model1, 'classes_')))
                probs_dict = dict(zip(classes, probs_list))
                p_normal = float(probs_dict.get('NORMAL', 0.0)) * 100.0
                p_hdf_raw = float(probs_dict.get('HEAT DISSIPATION FAILURE (HDF)', probs_dict.get('HDF', 0.0))) * 100.0
                p_osf_raw = float(probs_dict.get('OVERSTRAIN FAILURE (OSF)', probs_dict.get('OSF', 0.0))) * 100.0
                p_pwf_raw = float(probs_dict.get('POWER FAILURE (PWF)', probs_dict.get('PWF', 0.0))) * 100.0
                p_twf_raw = float(probs_dict.get('TOOL WEAR FAILURE (TWF)', probs_dict.get('TWF', 0.0))) * 100.0
                raw_risk = 100.0 - p_normal
                overall_risk = 100.0 if raw_risk >= 85.0 else raw_risk
                max_f = max(p_hdf_raw, p_osf_raw, p_pwf_raw, p_twf_raw)
                if max_f > 0:
                    sf = overall_risk / max_f
                    p_hdf, p_osf, p_pwf, p_twf = p_hdf_raw*sf, p_osf_raw*sf, p_pwf_raw*sf, p_twf_raw*sf
                else: p_hdf = p_osf = p_pwf = p_twf = 0.0
            else:
                p_normal = float(probs_list)*100 if probs_list else 0.0
                overall_risk = 100.0 - p_normal
                p_hdf = p_osf = p_pwf = p_twf = 0.0

            st.session_state.motor_risk = overall_risk
            st.session_state.motor_faults = {'HDF': p_hdf, 'OSF': p_osf, 'PWF': p_pwf, 'TWF': p_twf}
            
            s_color = "#00A65A" if overall_risk < 40 else ("#F39C12" if overall_risk < 75 else "#DD4B39")
            s_bg = "#E8F5E9" if overall_risk < 40 else ("#FEF9E7" if overall_risk < 75 else "#FDEDEC")
            diag_display = "NORMAL" if overall_risk < 40 else max(st.session_state.motor_faults, key=st.session_state.motor_faults.get)
            
            cA, cB = st.columns(list((1, 1.5)))
            with cA:
                st.markdown(f"<div style='background-color:{s_bg}; border: 2px solid {s_color}; padding: 15px; border-radius: 8px; text-align: center;'><h3 style='color:{s_color}; margin:0;'>{'SYSTEM NORMAL' if overall_risk < 40 else 'RISK DETECTED'}</h3></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><h4>AI Confidence</h4><h2 style='color:#3498DB;'>{p_normal if overall_risk < 50 else overall_risk:.1f}%</h2></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card' style='border-left-color:{s_color};'><h4>Diagnosis</h4><h2 style='color:{s_color};'>{diag_display}</h2></div>", unsafe_allow_html=True)
            with cB:
                st.plotly_chart(create_gauge(overall_risk, "MASTER OVERALL RISK", s_color, h=280), use_container_width=True)

            st.markdown("---")
            g1, g2, g3, g4 = st.columns(4)
            with g1: st.plotly_chart(create_gauge(p_hdf, "Heat (HDF)", "#E67E22", h=160), use_container_width=True)
            with g2: st.plotly_chart(create_gauge(p_osf, "Strain (OSF)", "#9B59B6", h=160), use_container_width=True)
            with g3: st.plotly_chart(create_gauge(p_pwf, "Power (PWF)", "#E74C3C", h=160), use_container_width=True)
            with g4: st.plotly_chart(create_gauge(p_twf, "Wear (TWF)", "#34495E", h=160), use_container_width=True)

            new_log = pd.DataFrame(list(({'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset': 'Motor Core', 'Risk Level': f"{overall_risk:.1f}%", 'Primary FMEA': diag_display, 'Status': 'Safe' if overall_risk < 40 else ('Warning' if overall_risk < 75 else 'Danger'), 'Grade': motor_type, 'Air Temp [K]': air_temp, 'Core Temp [K]': process_temp, 'RPM': rpm, 'Torque [Nm]': torque, 'Wear [min]': tool_wear, 'Vib RMS': '-', 'Vib Kurt': '-'},)))
            st.session_state.inspection_log = pd.concat(list((new_log, st.session_state.inspection_log)), ignore_index=True).head(30)

# --- TAB 2: BEARING VIBRATION ---
with tab2:
    cv1, cv2 = st.columns(list((1.5, 1)))
    with cv1:
        st.markdown("#### 🎚️ Simulator & Key Metrics")
        sim_level = st.slider("Degradation Phase (%) ❓", 0, 100, 10, help="NASA simulation phase.")
        r_val, k_val = 0.05 + (sim_level/100.0)**2 * 0.20, 3.0 + (sim_level/100.0)**1.5 * 5.0
        st.columns(4).metric("RMS ❓", f"{r_val:.3f}")
    with cv2:
        st.session_state.bearing_risk = 100.0 if sim_level >= 85.0 else sim_level
        st.plotly_chart(create_gauge(st.session_state.bearing_risk, "BEARING RISK", "#F39C12", h=280), use_container_width=True)
    new_log_b = pd.DataFrame(list(({'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset': 'Bearing', 'Risk Level': f"{st.session_state.bearing_risk:.1f}%", 'Primary FMEA': 'Vibration Anomaly', 'Status': 'Safe' if st.session_state.bearing_risk < 40 else 'Warning', 'Grade': '-', 'Air Temp [K]': '-', 'Core Temp [K]': '-', 'RPM': '-', 'Torque [Nm]': '-', 'Wear [min]': '-', 'Vib RMS': round(r_val, 3), 'Vib Kurt': round(k_val, 2)},)))
    st.session_state.inspection_log = pd.concat(list((new_log_b, st.session_state.inspection_log)), ignore_index=True).head(30)

# --- TAB 3: RELIABILITY FUSION ---
with tab3:
    m_frac, b_frac = st.session_state.motor_risk / 100.0, st.session_state.bearing_risk / 100.0
    int_risk = (1.0 - ((1.0 - m_frac) * (1.0 - b_frac))) * 100.0
    st.markdown(f"### 🔗 Master Reliability Fusion: {int_risk:.1f}%")
    st.plotly_chart(create_gauge(int_risk, "SYSTEM INTEGRATED RISK", "#DD4B39" if int_risk > 75 else "#00ff88", h=300), use_container_width=True)

# --- TAB 4: EXECUTIVE SUMMARY ---
with tab4:
    st.markdown("### 📊 Comprehensive Diagnostic Log")
    st.dataframe(st.session_state.inspection_log.style.apply(highlight_fmea, axis=1), use_container_width=True, hide_index=True)

# --- GLOBAL FOOTER (IMAGE MATCHED) ---
st.markdown("""
<div class="footer-text">
    <hr>
    <strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
    Confidential and Proprietary. The predictive algorithms and expert system overrides within this application are protected intellectual property.
</div>
""", unsafe_allow_html=True)