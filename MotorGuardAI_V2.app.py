# ==============================================================================
# © 2026 MotorGuard APM | Enterprise Predictive Maintenance System
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
st.set_page_config(page_title="MotorGuard APM", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 25px;
        border-radius: 12px;
        border-left: 8px solid #00ff88;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-top: 5px solid #3498DB;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'motor_risk' not in st.session_state: st.session_state.motor_risk = 0.0
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF':0.0, 'OSF':0.0, 'PWF':0.0, 'TWF':0.0}
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'authenticated' not in st.session_state: st.session_state.authenticated = False

# =========================================
# 2. AUTHENTICATION SYSTEM
# =========================================
def check_password():
    if st.session_state.authenticated:
        return True

    col1, col2, col3 = st.columns((1, 2, 1))
    with col2:
        st.markdown("""
        <div style="background-color:#F8F9FA; padding:40px; border-radius:15px; border-top: 5px solid #00ff88; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.1); margin-top: 50px;">
            <h2 style="color:#1E1E1E; margin-bottom: 0px;">MotorGuard <span style="color:#00ff88;">APM</span></h2>
            <p style="color:#666; font-size: 14px; margin-bottom: 25px;">Secure Enterprise Diagnostics Portal</p>
        """, unsafe_allow_html=True)
        
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Log In to System", use_container_width=True, type="primary"):
            if u == "admin" and p == "DBA2026":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Access Denied: Invalid Credentials")
        
        st.markdown("""
            <hr>
            <p style="font-size:11px; color:#888; line-height:1.4;">
                © 2026 MotorGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.<br>
                Confidential and Proprietary Intellectual Property.
            </p>
        </div>
        """, unsafe_allow_html=True)
    return False

if not check_password():
    st.stop()

# =========================================
# 3. AI MODELS LOADING
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
# 4. SIDEBAR & LOGOUT
# =========================================
with st.sidebar:
    st.markdown("### 🛠️ Control Panel")
    if st.button("🚪 Terminate Session", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    st.markdown("---")
    st.caption("Connected to Edge Sensors...")

# =========================================
# 5. MAIN INTERFACE
# =========================================
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin:0;">🏭 MotorGuard APM <span style="color:#00ff88;">System</span></h1>
            <p style="margin:0; font-size: 14px; color:#e0e0e0;">Predictive Maintenance Dashboard | Eng. Mahmoud Reda Ibrahim</p>
        </div>
        <div style="text-align: right;">
            <p style="margin:0; color:#00ff88; font-weight: bold;">⚡ LIVE TELEMETRY</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs(["⚙️ Motor Health", "📳 Bearing Vibration", "🔗 System Fusion", "📊 Executive Summary"])

# --- TAB 1: MOTOR ---
with t1:
    col_in, col_out = st.columns((1, 2))
    with col_in:
        m_grade = st.selectbox("Motor Quality", ("L (Low)", "M (Medium)", "H (High)"), index=1)
        air_v = st.number_input("Ambient Temp [K]", 290.0, 310.0, 298.0)
        proc_v = st.number_input("Core Temp [K]", 300.0, 320.0, 308.0)
        rpm_v = st.slider("Rotational Speed [RPM]", 1100, 2900, 1500)
        trq_v = st.slider("Torque Load [Nm]", 10.0, 80.0, 40.0)
        wear_v = st.slider("Cumulative Wear [min]", 0, 260, 50)
    
    with col_out:
        if models_ready:
            g_val = 0 if "L" in m_grade else (1 if "M" in m_grade else 2)
            # تم تصحيح أسماء الأعمدة لتطابق الـ Scaler الأصلي
            d1 = pd.DataFrame({
                'Type': [g_val],
                'Air temperature [K]': [air_v],
                'Process temperature [K]': [proc_v],
                'Rotational speed [rpm]': [rpm_v],
                'Torque [Nm]': [trq_v],
                'Tool wear [min]': [wear_v]
            })
            d1_sc = scaler1.transform(d1) # سيتم التحويل بنجاح الآن
            p1 = np.array(model1.predict_proba(d1_sc)).flatten()
            
            f_hdf = float(p1) * 100 if len(p1) > 1 else 0.0
            f_osf = float(p1) * 100 if len(p1) > 2 else 0.0
            f_pwf = float(p1) * 100 if len(p1) > 3 else 0.0
            f_twf = float(p1) * 100 if len(p1) > 4 else 0.0
            
            m_risk = (1.0 - ((1.0-f_hdf/100)*(1.0-f_osf/100)*(1.0-f_pwf/100)*(1.0-f_twf/100))) * 100
            st.session_state.motor_risk = m_risk
            st.session_state.motor_faults = {'HDF':f_hdf, 'OSF':f_osf, 'PWF':f_pwf, 'TWF':f_twf}
            
            fig1 = go.Figure(go.Indicator(mode="gauge+number", value=m_risk, title={'text': "Motor Overall Risk"}, gauge={'axis':{'range':(0,100)}, 'bar':{'color':"#3498DB"}}))
            st.plotly_chart(fig1, use_container_width=True)

# --- TAB 2: BEARING ---
with t2:
    col_v1, col_v2 = st.columns((1, 2))
    with col_v1:
        st.info("NASA Vibration Run-to-Failure Simulator")
        deg = st.slider("Bearing Degradation Phase", 0, 100, 10)
        r = 0.05 + (deg/100.0)**2 * 0.20
        k = 3.0 + (deg/100.0)**1.5 * 5.0
        sk = (deg/100.0) * 0.5
        p2p = r * 4.0
        cr = 3.5 + (deg/100.0)
    with col_v2:
        if models_ready:
            d2 = pd.DataFrame({'RMS': [r], 'Kurtosis': [k], 'Skewness': [sk], 'Peak_to_Peak': [p2p], 'Crest_Factor': [cr]})
            d2_sc = scaler2.transform(d2)
            p2 = np.array(model2.predict_proba(d2_sc)).flatten()
            
            b_risk = (float(p2)*0.4 + float(p2)*0.8 + float(p2)*1.0) * 100 if len(p2) > 1 else 0.0
            st.session_state.bearing_risk = b_risk
            
            fig2 = go.Figure(go.Indicator(mode="gauge+number", value=b_risk, title={'text': "Bearing Risk Status"}, gauge={'axis':{'range':(0,100)}, 'bar':{'color':"#9B59B6"}}))
            st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: FUSION ---
with t3:
    m_f = st.session_state.motor_risk / 100.0
    b_f = st.session_state.bearing_risk / 100.0
    total_f = (1.0 - ((1.0-m_f)*(1.0-b_f))) * 100.0
    
    st.markdown("### 🔗 System Reliability Analysis")
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=total_f, title={'text': "INTEGRATED SYSTEM RISK"}, gauge={'axis':{'range':(0,100)}, 'bar':{'color':"#E74C3C"}})), use_container_width=True)

# --- TAB 4: EXECUTIVE ---
with t4:
    total_risk = (1.0 - ((1.0-st.session_state.motor_risk/100.0)*(1.0-st.session_state.bearing_risk/100.0))) * 100.0
    h_score = 100.0 - total_risk
    
    ex1, ex2, ex3 = st.columns(3)
    ex1.metric("Overall Asset Health", f"{h_score:.1f}%")
    ex2.metric("Downtime Risk", f"{total_risk:.1f}%")
    ex3.metric("Status", "DANGER" if total_risk > 75 else ("STABLE" if total_risk < 40 else "WARNING"))
    
    st.markdown("#### Holistic Risk Distribution Radar")
    rv = [st.session_state.motor_faults['HDF'], st.session_state.motor_faults['OSF'], st.session_state.motor_faults['PWF'], st.session_state.motor_faults['TWF'], st.session_state.bearing_risk]
    ct = ['Heat', 'Strain', 'Power', 'Wear', 'Vibration']
    fig_radar = go.Figure(go.Scatterpolar(r=rv, theta=ct, fill='toself', fillcolor='rgba(231, 76, 60, 0.3)', line=dict(color='#E74C3C')))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=(0,100))), showlegend=False)
    st.plotly_chart(fig_radar, use_container_width=True)

# Footer
st.markdown("""
<br><hr>
<div style='text-align: center; color: #888888; font-size: 11px;'>
    <strong>© 2026 MotorGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
    Confidential and Proprietary Intellectual Property.
</div>
""", unsafe_allow_html=True)