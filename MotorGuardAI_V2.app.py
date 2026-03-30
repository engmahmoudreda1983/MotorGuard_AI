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
    </style>
""", unsafe_allow_html=True)

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if 'motor_risk' not in st.session_state: st.session_state.motor_risk = 0.0
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF': 0.0, 'OSF': 0.0, 'PWF': 0.0, 'TWF': 0.0}

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
    except Exception as e:
        return None, None, None, None, False

scaler1, model1, scaler2, model2, models_ready = load_ai_assets()

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
            'axis': {'range': (0, max_val), 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color_theme},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': (
                {'range': (0, 40), 'color': '#E8F5E9'},  
                {'range': (40, 75), 'color': '#FEF9E7'},  
                {'range': (75, 100), 'color': '#FDEDEC'}  
            ),
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

# --- Sidebar Control ---
with st.sidebar:
    st.markdown("### 🖥️ Dashboard Control")
    if st.button("🚪 Secure Logout", use_container_width=True):
        logout()
    st.markdown("---")
    if not models_ready:
        st.error("⚠️ AI Models not found.")
    else:
        st.success("✅ Edge Sensors Connected.")

# --- Header ---
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin:0;">⚙️ MotorGuard AI <span style="color:#DD4B39;">Pro</span></h2>
            <p style="margin:0; font-size: 14px; color:#CCCCCC;">Enterprise Asset Performance Management</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- TABS SETUP ---
tab1, tab2, tab3, tab4 = st.tabs((
    "⚙️ Model 1: Motor Diagnostics", 
    "📳 Model 2: NASA Bearing", 
    "🔗 System Reliability Fusion", 
    "📊 Executive Summary"
))

# =========================================
# TAB 1: MOTOR DIAGNOSTICS
# =========================================
with tab1:
    st.markdown("### ⚙️ Thermomechanical Fault Detection (AI4I)")
    
    col_input, col_dashboard = st.columns((1, 2.5))
    
    with col_input:
        st.markdown("#### 🎛️ Telemetry")
        motor_options = {'L (Low)': 0, 'M (Medium)': 1, 'H (High)': 2}
        motor_type = st.selectbox("Motor Quality Grade", list(motor_options.keys()), index=1)
        air_temp = st.number_input("Ambient Temp [K]", 290.0, 310.0, 298.0, 0.1)
        process_temp = st.number_input("Motor Core Temp [K]", 300.0, 320.0, 308.0, 0.1)
        rpm = st.slider("Rotational Speed [RPM]", 1100, 2900, 1500, 10)
        torque = st.slider("Applied Torque [Nm]", 10.0, 80.0, 40.0, 0.5)
        tool_wear = st.slider("Tool Wear [min]", 0, 260, 50, 1)

    with col_dashboard:
        if models_ready:
            motor_cols = ('Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]')
            scale_cols = ('Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]')
            
            # Using tuple of tuples to completely avoid list filtering bugs
            d1_values = ((motor_options[motor_type], air_temp, process_temp, rpm, torque, tool_wear),)
            motor_df = pd.DataFrame(d1_values, columns=motor_cols)
            
            scaled_features = scaler1.transform(motor_df[list(scale_cols)])
            motor_df[list(scale_cols)] = scaled_features
            
            probs_matrix = model1.predict_proba(motor_df)
            probs_list = np.array(probs_matrix).flatten().tolist()
            
            p_normal = probs_list * 100.0 if len(probs_list) > 0 else 100.0
            p_hdf_raw = probs_list * 100.0 if len(probs_list) > 1 else 0.0
            p_osf_raw = probs_list * 100.0 if len(probs_list) > 2 else 0.0
            p_pwf_raw = probs_list * 100.0 if len(probs_list) > 3 else 0.0
            p_twf_raw = probs_list * 100.0 if len(probs_list) > 4 else 0.0

            # Expert System
            expert_twf = min(50.0 + ((tool_wear - 200) / 40.0) * 45.0, 98.0) if tool_wear >= 200 else 0.0
            power = torque * rpm
            if power > 75000: expert_pwf = min(40.0 + ((power - 75000) / 15000.0) * 55.0, 98.0)
            elif power < 40000: expert_pwf = min(40.0 + ((40000 - power) / 10000.0) * 55.0, 98.0)
            else: expert_pwf = 0.0
            strain = torque * tool_wear
            strain_limit = 11000 if motor_options[motor_type] == 0 else (12000 if motor_options[motor_type] == 1 else 13000)
            warning_limit = strain_limit * 0.8
            expert_osf = min(40.0 + ((strain - warning_limit) / (strain_limit - warning_limit)) * 55.0, 98.0) if strain > warning_limit else 0.0
            temp_diff = process_temp - air_temp
            expert_hdf = min(40.0 + ((10.0 - temp_diff) / 2.0) * 55.0, 98.0) if temp_diff <= 10.0 and rpm < 1500 else 0.0

            # Merge
            p_hdf = max(p_hdf_raw, expert_hdf)
            p_osf = max(p_osf_raw, expert_osf)
            p_pwf = max(p_pwf_raw, expert_pwf)
            p_twf = max(p_twf_raw, expert_twf)

            # Cumulative
            p_survive = (1.0 - p_hdf/100.0) * (1.0 - p_osf/100.0) * (1.0 - p_pwf/100.0) * (1.0 - p_twf/100.0)
            calc_risk = (1.0 - p_survive) * 100.0
            overall_motor_risk = min(max((100.0 - p_normal), calc_risk), 100.0)
            
            st.session_state.motor_risk = overall_motor_risk
            st.session_state.motor_faults = {'HDF': p_hdf, 'OSF': p_osf, 'PWF': p_pwf, 'TWF': p_twf}

            status_color = "#00A65A" if overall_motor_risk < 40 else ("#F39C12" if overall_motor_risk < 75 else "#DD4B39")
            
            c_gauge, c_bars = st.columns((1, 1.5))
            with c_gauge:
                st.plotly_chart(create_industrial_gauge(overall_motor_risk, "MOTOR RISK", status_color, height=250), use_container_width=True)
            with c_bars:
                g1, g2 = st.columns(2)
                g1.plotly_chart(create_industrial_gauge(p_hdf, "Heat (HDF)", "#E67E22", height=150), use_container_width=True)
                g2.plotly_chart(create_industrial_gauge(p_osf, "Strain (OSF)", "#9B59B6", height=150), use_container_width=True)
                g1.plotly_chart(create_industrial_gauge(p_pwf, "Power (PWF)", "#E74C3C", height=150), use_container_width=True)
                g2.plotly_chart(create_industrial_gauge(p_twf, "Wear (TWF)", "#34495E", height=150), use_container_width=True)

# =========================================
# TAB 2: BEARING VIBRATION (NASA)
# =========================================
with tab2:
    st.markdown("### 📳 High-Frequency Vibration Analysis")
    
    cv1, cv2 = st.columns((1, 2.5))
    with cv1:
        st.markdown("#### 🎚️ Simulator")
        st.info("Simulating Run-to-Failure degradation.")
        sim_level = st.slider("Degradation Phase (%)", 0, 100, 10)
        
        r_val = 0.05 + (sim_level/100.0)**2 * 0.20
        k_val = 3.0 + (sim_level/100.0)**1.5 * 5.0
        sk_val = (sim_level/100.0) * 0.5
        p2p_val = r_val * 4.5
        cr_val = 3.5 + (sim_level/100.0)
        
        st.caption(f"**RMS:** {r_val:.3f} | **Kurt:** {k_val:.2f}")

    with cv2:
        if models_ready:
            bearing_cols = ('RMS', 'Kurtosis', 'Skewness', 'Peak_to_Peak', 'Crest_Factor')
            d2_values = ((r_val, k_val, sk_val, p2p_val, cr_val),)
            bearing_df = pd.DataFrame(d2_values, columns=bearing_cols)
            
            scaled_b = scaler2.transform(bearing_df)
            probs_b = np.array(model2.predict_proba(scaled_b)).flatten().tolist()
            
            p_b_early = probs_b * 100.0 if len(probs_b) > 1 else 0.0
            p_b_warn = probs_b * 100.0 if len(probs_b) > 2 else 0.0
            p_b_fail = probs_b * 100.0 if len(probs_b) > 3 else 0.0
            
            b_risk = (p_b_early * 0.4) + (p_b_warn * 0.8) + (p_b_fail * 1.0)
            st.session_state.bearing_risk = b_risk
            
            b_color = "#00A65A" if b_risk < 40 else ("#F39C12" if b_risk < 75 else "#DD4B39")
            
            cb1, cb2 = st.columns(2)
            with cb1:
                st.plotly_chart(create_industrial_gauge(b_risk, "BEARING VIBRATION RISK", b_color, height=280), use_container_width=True)
            with cb2:
                df_b_faults = pd.DataFrame({'Phase': ('Early Wear', 'Warning', 'Failure'), 'Probability': (p_b_early, p_b_warn, p_b_fail)})
                fig_b = px.bar(df_b_faults, x='Probability', y='Phase', orientation='h', color='Probability', color_continuous_scale='Purples', range_x=(0,100))
                fig_b.update_layout(height=280, margin=dict(l=0,r=0,t=30,b=0), title="AI Confidence")
                st.plotly_chart(fig_b, use_container_width=True)

# =========================================
# TAB 3: RELIABILITY FUSION
# =========================================
with tab3:
    st.markdown("### 🔗 Master System Fusion (Series Configuration)")
    
    m_frac = st.session_state.motor_risk / 100.0
    b_frac = st.session_state.bearing_risk / 100.0
    integrated_risk = (1.0 - ((1.0 - m_frac) * (1.0 - b_frac))) * 100.0
    
    f_color = "#00A65A" if integrated_risk < 40 else ("#F39C12" if integrated_risk < 75 else "#DD4B39")
    
    cf1, cf2, cf3 = st.columns((1, 0.2, 1))
    with cf1:
        st.markdown(f"<div class='metric-card' style='border-left-color:#3498DB;'><h4>Core Motor Risk</h4><h1 style='color:#3498DB;'>{st.session_state.motor_risk:.1f}%</h1></div>", unsafe_allow_html=True)
    with cf2:
        st.markdown("<h1 style='text-align:center; margin-top:30px; color:#aaa;'>✖</h1>", unsafe_allow_html=True)
    with cf3:
        st.markdown(f"<div class='metric-card' style='border-left-color:#9B59B6;'><h4>Bearing Vib Risk</h4><h1 style='color:#9B59B6;'>{st.session_state.bearing_risk:.1f}%</h1></div>", unsafe_allow_html=True)
    
    st.plotly_chart(create_industrial_gauge(integrated_risk, "OVERALL MACHINE RISK", f_color, height=300), use_container_width=True)

# =========================================
# TAB 4: EXECUTIVE SUMMARY
# =========================================
with tab4:
    st.markdown("### 📊 C-Level Executive Dashboard")
    
    health_score = 100.0 - integrated_risk
    
    ex1, ex2, ex3 = st.columns(3)
    ex1.metric("Asset Health Score", f"{health_score:.1f}%", delta="-Critical" if health_score < 40 else "Nominal", delta_color="inverse")
    ex2.metric("Unplanned Downtime Prob", f"{integrated_risk:.1f}%")
    
    status_text = "EMERGENCY SHUTDOWN" if integrated_risk > 75 else ("SCHEDULE MAINTENANCE" if integrated_risk > 40 else "OPTIMAL PRODUCTION")
    st.markdown(f"""
    <div style="background-color:#F8F9FA; padding:15px; border-radius:8px; border-left: 5px solid {f_color};">
        <h4 style="margin:0; color:#666;">AI Recommended Action</h4>
        <h3 style="margin:0; color:{f_color};">{status_text}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 🎯 Holistic Fault Distribution Radar")
    
    categories = ('Heat (HDF)', 'Strain (OSF)', 'Power (PWF)', 'Tool Wear (TWF)', 'Vibration (Bearing)')
    values = (
        st.session_state.motor_faults['HDF'],
        st.session_state.motor_faults['OSF'],
        st.session_state.motor_faults['PWF'],
        st.session_state.motor_faults['TWF'],
        st.session_state.bearing_risk
    )
    
    fig_radar = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', line_color='#DD4B39'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=(0, 100))), showlegend=False, height=450)
    st.plotly_chart(fig_radar, use_container_width=True)

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