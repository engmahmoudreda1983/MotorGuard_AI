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
st.set_page_config(page_title="MotorGuard APM Command Center", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

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
        transition: transform 0.3s;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'motor_risk' not in st.session_state: st.session_state.motor_risk = 0.0
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF':0.0, 'OSF':0.0, 'PWF':0.0, 'TWF':0.0}
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'authenticated' not in st.session_state: st.session_state.authenticated = False

# =========================================
# 2. AUTHENTICATION SYSTEM (First Priority)
# =========================================
def check_password():
    if st.session_state.authenticated:
        return True

    c1, c2, c3 = st.columns((1, 2, 1))
    with c2:
        st.markdown("<div style='background-color:#F8F9FA; padding:30px; border-radius:15px; border-top: 5px solid #00ff88; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 50px;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='color:#1E1E1E;'>MotorGuard <span style='color:#00ff88;'>APM</span></h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#666;'>Secure Enterprise Login</p>", unsafe_allow_html=True)
        
        u = st.text_input("Username", key="u_field")
        p = st.text_input("Password", type="password", key="p_field")
        
        if st.button("Enter Dashboard", use_container_width=True, type="primary"):
            if u == "admin" and p == "DBA2026":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Invalid Credentials")
        st.markdown("</div>", unsafe_allow_html=True)
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
# 4. DASHBOARD HEADER & SIDEBAR
# =========================================
st.sidebar.markdown("### 🖥️ Menu")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.authenticated = False
    st.rerun()

st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin:0;">🏭 MotorGuard APM <span style="color:#00ff88;">Enterprise</span></h1>
            <p style="margin:0; font-size: 14px; color:#e0e0e0;">Unified Asset Performance Management | Eng. Mahmoud Reda Ibrahim</p>
        </div>
        <div style="text-align: right;">
            <p style="margin:0; color:#00ff88; font-weight: bold;">STATUS: CONNECTED</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# تفعيل الـ Tabs بأسماء صريحة لتجنب أخطاء الـ Tuple والـ Context Manager
t1, t2, t3, t4 = st.tabs(["⚙️ Motor Health", "📳 Bearing Vibration", "🔗 System Fusion", "📊 Executive Summary"])

# --- TAB 1: MOTOR ---
with t1:
    col_in, col_out = st.columns((1, 2))
    with col_in:
        m_grade = st.selectbox("Grade", ("L", "M", "H"), index=1)
        air_v = st.number_input("Air Temp", 290.0, 310.0, 298.0)
        proc_v = st.number_input("Core Temp", 300.0, 320.0, 308.0)
        rpm_v = st.slider("RPM", 1100, 2900, 1500)
        trq_v = st.slider("Torque", 10.0, 80.0, 40.0)
        wear_v = st.slider("Wear", 0, 260, 50)
    
    with col_out:
        if models_ready:
            tg = 0 if m_grade=="L" else (1 if m_grade=="M" else 2)
            d1 = pd.DataFrame({'Type': (tg,), 'Air temperature [K]': (air_v,), 'Process temperature [K]': (proc_v,), 'Rotational speed [rpm]': (rpm_v,), 'Torque [Nm]': (trq_v,), 'Tool wear [min]': (wear_v,)})
            d1_sc = scaler1.transform(d1)
            p1 = np.array(model1.predict_proba(d1_sc)).flatten()
            
            # استخراج الاحتمالات بأمان
            f_hdf = float(p1) * 100 if len(p1) > 1 else 0.0
            f_osf = float(p1) * 100 if len(p1) > 2 else 0.0
            f_pwf = float(p1) * 100 if len(p1) > 3 else 0.0
            f_twf = float(p1) * 100 if len(p1) > 4 else 0.0
            
            m_risk = (1.0 - ((1.0-f_hdf/100)*(1.0-f_osf/100)*(1.0-f_pwf/100)*(1.0-f_twf/100))) * 100
            st.session_state.motor_risk = m_risk
            st.session_state.motor_faults = {'HDF':f_hdf, 'OSF':f_osf, 'PWF':f_pwf, 'TWF':f_twf}
            
            st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=m_risk, title={'text': "Motor Overall Risk"}, gauge={'axis':{'range':(0,100)}, 'bar':{'color':"#3498DB"}})), use_container_width=True)

# --- TAB 2: BEARING ---
with t2:
    col_v1, col_v2 = st.columns((1, 2))
    with col_v1:
        st.info("NASA Vibration Simulator")
        deg = st.slider("Degradation Level", 0, 100, 10)
        r = 0.05 + (deg/100.0)**2 * 0.20
        k = 3.0 + (deg/100.0)**1.5 * 5.0
    with col_v2:
        if models_ready:
            # استخدام قيم صريحة لمنع حذف المصفوفات
            d2 = pd.DataFrame({'RMS': (r,), 'Kurtosis': (k,), 'Skewness': (0.0,), 'Peak_to_Peak': (0.0,), 'Crest_Factor': (0.0,)})
            d2_sc = scaler2.transform(d2)
            p2 = np.array(model2.predict_proba(d2_sc)).flatten()
            
            b_risk = (float(p2)*0.4 + float(p2)*0.8 + float(p2)*1.0) * 100 if len(p2) > 1 else 0.0
            st.session_state.bearing_risk = b_risk
            
            st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=b_risk, title={'text': "Bearing Risk"}, gauge={'axis':{'range':(0,100)}, 'bar':{'color':"#9B59B6"}})), use_container_width=True)

# --- TAB 3: FUSION ---
with t3:
    m_f = st.session_state.motor_risk / 100.0
    b_f = st.session_state.bearing_risk / 100.0
    combined = (1.0 - ((1.0-m_f)*(1.0-b_f))) * 100.0
    
    st.markdown("### 🔗 System Reliability Fusion")
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=combined, title={'text': "INTEGRATED RISK"}, gauge={'axis':{'range':(0,100)}, 'bar':{'color':"#E74C3C"}})), use_container_width=True)

# --- TAB 4: EXECUTIVE ---
with t4:
    total_f = (1.0 - ((1.0-st.session_state.motor_risk/100.0)*(1.0-st.session_state.bearing_risk/100.0))) * 100.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Asset Health", f"{100-total_f:.1f}%")
    c2.metric("Downtime Prob", f"{total_f:.1f}%")
    c3.metric("Status", "DANGER" if total_f > 75 else ("WARNING" if total_f > 40 else "SAFE"))
    
    st.markdown("#### Risk Distribution Radar")
    rv = [st.session_state.motor_faults['HDF'], st.session_state.motor_faults['OSF'], st.session_state.motor_faults['PWF'], st.session_state.motor_faults['TWF'], st.session_state.bearing_risk]
    ct = ['Heat', 'Strain', 'Power', 'Wear', 'Vibration']
    fig_radar = go.Figure(go.Scatterpolar(r=rv, theta=ct, fill='toself'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=(0,100))))
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("<br><hr><center><p style='color:#888;'>© 2026 MotorGuard APM | Eng. Mahmoud Reda Ibrahim</p></center>", unsafe_allow_html=True)