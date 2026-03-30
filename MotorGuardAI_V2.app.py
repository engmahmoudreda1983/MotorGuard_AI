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
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF':0, 'OSF':0, 'PWF':0, 'TWF':0}
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'bearing_status' not in st.session_state: st.session_state.bearing_status = "Normal"
if 'authenticated' not in st.session_state: st.session_state.authenticated = False

# =========================================
# 2. AUTHENTICATION SYSTEM
# =========================================
def check_password():
    if st.session_state["authenticated"]:
        return True

    col1, col2, col3 = st.columns((1.5, 2.0, 1.5)) 
    with col2: 
        st.markdown("<div style='background-color:#F8F9FA; padding:30px; border-radius:15px; border-top: 5px solid #00ff88; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 50px;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='color:#1E1E1E; margin-bottom: 5px;'>MotorGuard <span style='color:#00ff88;'>APM</span></h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#666; font-size: 14px;'>Secure Enterprise Login</p>", unsafe_allow_html=True)
        
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Access Dashboard", use_container_width=True, type="primary"):
            if user == "admin" and password == "DBA2026":
                st.session_state["authenticated"] = True
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
        scaler1 = joblib.load('scaler.pkl')
        model1 = joblib.load('motorguard_model.pkl')
        scaler2 = joblib.load('nasa_bearing_scaler.pkl')
        model2 = joblib.load('nasa_bearing_model.pkl')
        return scaler1, model1, scaler2, model2, True
    except:
        return None, None, None, None, False

scaler1, model1, scaler2, model2, models_loaded = load_ai_assets()

# =========================================
# 4. DASHBOARD UI
# =========================================
st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin:0;">🏭 MotorGuard APM <span style="color:#00ff88;">Enterprise</span></h1>
            <p style="margin:0; font-size: 14px; color:#e0e0e0;">Developed by Eng. Mahmoud Reda Ibrahim</p>
        </div>
        <div style="text-align: right;">
            <p style="margin:0; color:#00ff88; font-weight: bold;">STATUS: CONNECTED</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(("⚙️ Motor Health", "📳 Bearing Vibration", "🔗 System Fusion", "📊 Executive Summary"))

# --- TAB 1: MOTOR ---
with tab1:
    col_in, col_out = st.columns((1, 2))
    with col_in:
        m_type = st.selectbox("Grade", ('L', 'M', 'H'), index=1)
        air_t = st.number_input("Air Temp", 290.0, 310.0, 298.0)
        proc_t = st.number_input("Proc Temp", 300.0, 320.0, 308.0)
        rpm = st.slider("RPM", 1100, 2900, 1500)
        trq = st.slider("Torque", 10.0, 80.0, 40.0)
        wear = st.slider("Wear", 0, 260, 50)
    
    with col_out:
        if models_loaded:
            t_val = 0 if m_type=='L' else (1 if m_type=='M' else 2)
            d1 = pd.DataFrame({'Type':[t_val], 'Air temperature [K]':[air_t], 'Process temperature [K]':[proc_t], 'Rotational speed [rpm]':[rpm], 'Torque [Nm]':[trq], 'Tool wear [min]':[wear]})
            d1_sc = scaler1.transform(d1)
            p1_raw = model1.predict_proba(d1_sc)
            p1 = np.array(p1_raw).flatten()
            
            # Safe Map
            p_hdf = float(p1)*100 if len(p1)>1 else 0
            p_osf = float(p1)*100 if len(p1)>2 else 0
            p_pwf = float(p1)*100 if len(p1)>3 else 0
            p_twf = float(p1)*100 if len(p1)>4 else 0
            
            risk1 = (1.0 - ((1-p_hdf/100)*(1-p_osf/100)*(1-p_pwf/100)*(1-p_twf/100))) * 100
            st.session_state.motor_risk = risk1
            st.session_state.motor_faults = {'HDF':p_hdf, 'OSF':p_osf, 'PWF':p_pwf, 'TWF':p_twf}
            
            fig1 = go.Figure(go.Indicator(mode="gauge+number", value=risk1, title={'text': "Motor Risk"}, gauge={'axis':{'range':(0,100)}, 'bar':{'color':"#3498DB"}}))
            st.plotly_chart(fig1, use_container_width=True)

# --- TAB 2: BEARING ---
with tab2:
    col_v1, col_v2 = st.columns((1, 2))
    with col_v1:
        sim = st.slider("Degradation (%)", 0, 100, 10)
        r_v = 0.05 + (sim/100)**2 * 0.20
        k_v = 3.0 + (sim/100)**1.5 * 5.0
    with col_v2:
        if models_loaded:
            d2 = pd.DataFrame({'RMS':[r_v], 'Kurtosis':[k_v], 'Skewness':, 'Peak_to_Peak':, 'Crest_Factor':})
            d2_sc = scaler2.transform(d2)
            p2_raw = model2.predict_proba(d2_sc)
            p2 = np.array(p2_raw).flatten()
            
            risk2 = (float(p2)*0.4 + float(p2)*0.8 + float(p2)*1.0) * 100 if len(p2)>1 else 0
            st.session_state.bearing_risk = risk2
            
            fig2 = go.Figure(go.Indicator(mode="gauge+number", value=risk2, title={'text': "Bearing Risk"}, gauge={'axis':{'range':(0,100)}, 'bar':{'color':"#9B59B6"}}))
            st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: FUSION ---
with tab3:
    m_f = st.session_state.motor_risk / 100
    b_f = st.session_state.bearing_risk / 100
    total_f = (1.0 - ((1-m_f)*(1-b_f))) * 100
    
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=total_f, title={'text': "MASTER ASSET RISK"}, gauge={'axis':{'range':(0,100)}, 'bar':{'color':"#E74C3C"}})), use_container_width=True)

# --- TAB 4: EXECUTIVE ---
with tab4:
    h_score = 100 - total_f
    c1, c2, c3 = st.columns(3)
    c1.metric("Health Score", f"{h_score:.1f}%")
    c2.metric("Downtime Prob", f"{total_f:.1f}%")
    c3.metric("Status", "CRITICAL" if total_f>75 else "STABLE")
    
    st.markdown("#### Holistic Risk Profile")
    vals = [st.session_state.motor_faults['HDF'], st.session_state.motor_faults['OSF'], st.session_state.motor_faults['PWF'], st.session_state.motor_faults['TWF'], st.session_state.bearing_risk]
    cats = ['Heat', 'Strain', 'Power', 'Wear', 'Vibration']
    fig_r = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself'))
    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=(0,100))))
    st.plotly_chart(fig_r, use_container_width=True)

# --- LOGOUT ---
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout Session"):
    st.session_state.authenticated = False
    st.rerun()