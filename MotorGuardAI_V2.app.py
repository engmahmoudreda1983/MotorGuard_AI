# ==============================================================================
# © 2026 MotorGuard AI | Enterprise Predictive Maintenance & FMEA
# Developed by: Eng. Mahmoud Reda Ibrahim (DBA Research)
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
    .fmea-table { font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

# --- Session States ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'motor_risk' not in st.session_state: st.session_state.motor_risk = 0.0
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF': 0.0, 'OSF': 0.0, 'PWF': 0.0, 'TWF': 0.0}
if 'bearing_phase' not in st.session_state: st.session_state.bearing_phase = "Normal"
if 'inspection_log' not in st.session_state:
    st.session_state['inspection_log'] = pd.DataFrame(columns=['Timestamp', 'Asset', 'Risk Level', 'Primary FMEA', 'Status'])

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
    if st.session_state.authenticated: return True
    col1, col2, col3 = st.columns((1.5, 2.0, 1.5)) 
    with col2: 
        st.markdown("""
        <div style="background-color:#F8F9FA; padding:30px; border-radius:15px; border-top: 5px solid #DD4B39; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 50px;">
            <h2 style="color:#1E1E1E; margin-bottom: 5px;">MotorGuard <span style="color:#DD4B39;">AI</span></h2>
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
        st.markdown("<hr><div style='text-align: center; color: #888; font-size: 11px;'><strong>© 2026 MotorGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>Confidential and Proprietary.</div>", unsafe_allow_html=True)
    return False

def create_gauge(value, title, color, max_v=100, h=250):
    val = float(value) if not pd.isna(value) else 0.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number={'suffix': "%", 'font': {'size': 24, 'color': color}},
        title={'text': title, 'font': {'size': 14, 'color': '#555'}},
        gauge={
            'axis': {'range': (0, max_v), 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
            'steps': ({'range': (0, 40), 'color': '#E8F5E9'}, {'range': (40, 75), 'color': '#FEF9E7'}, {'range': (75, 100), 'color': '#FDEDEC'}),
            'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': val}
        }))
    fig.update_layout(height=h, margin=dict(l=15, r=15, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def highlight_fmea(row):
    status = row.get('Status', '')
    if status == 'Safe': return ['background-color: #E8F5E9; color: #00A65A; font-weight: bold'] * len(row)
    if status == 'Warning': return ['background-color: #FEF9E7; color: #D35400; font-weight: bold'] * len(row)
    if status == 'Danger': return ['background-color: #FDEDEC; color: #C0392B; font-weight: bold'] * len(row)
    return [''] * len(row)

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
    else: st.success("✅ Sensors Connected.")

st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin:0;">⚙️ MotorGuard AI <span style="color:#DD4B39;">Pro</span></h2>
            <p style="margin:0; font-size: 14px; color:#CCCCCC;">Enterprise Asset Performance Management</p>
        </div>
        <div style="text-align: right;"><h4 style="margin:0; color:#00ff88;">⚡ LIVE</h4></div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(("⚙️ Motor Diagnostics", "📳 Bearing Vibration", "🔗 Reliability Fusion", "📊 Executive Summary"))

# --- TAB 1: MOTOR DIAGNOSTICS ---
with tab1:
    c_in, c_out = st.columns((1, 2.5))
    with c_in:
        st.markdown("#### 🎛️ Telemetry Input")
        motor_options = {'L (Low)': 0, 'M (Medium)': 1, 'H (High)': 2}
        motor_type = st.selectbox("Motor Quality", list(motor_options.keys()), index=1)
        air_temp = st.number_input("Ambient Temp [K]", 290.0, 310.0, 298.0, 0.1)
        process_temp = st.number_input("Core Temp [K]", 300.0, 320.0, 308.0, 0.1)
        rpm = st.slider("Speed [RPM]", 1100, 2900, 1500, 10)
        torque = st.slider("Torque [Nm]", 10.0, 80.0, 40.0, 0.5)
        tool_wear = st.slider("Tool Wear [min]", 0, 260, 50, 1)
        analyze_btn = st.button("🔍 Analyze Motor", use_container_width=True, type="primary")

    with c_out:
        if models_ready:
            c_air = 'Air temperature ' + chr(91) + 'K' + chr(93)
            c_proc = 'Process temperature ' + chr(91) + 'K' + chr(93)
            c_rpm = 'Rotational speed ' + chr(91) + 'rpm' + chr(93)
            c_trq = 'Torque ' + chr(91) + 'Nm' + chr(93)
            c_wear = 'Tool wear ' + chr(91) + 'min' + chr(93)
            motor_cols = ('Type', c_air, c_proc, c_rpm, c_trq, c_wear)
            scale_cols = (c_air, c_proc, c_rpm, c_trq, c_wear)
            
            d1_vals = ((motor_options.get(motor_type), air_temp, process_temp, rpm, torque, tool_wear),)
            motor_df = pd.DataFrame(d1_vals, columns=motor_cols)
            motor_df[list(scale_cols)] = scaler1.transform(motor_df[list(scale_cols)])
            
            # --- Robust V2 Probability Extraction ---
            probs_matrix = model1.predict_proba(motor_df)
            probs_list = np.array(probs_matrix).flatten().tolist()
            
            p_normal = p_hdf_raw = p_osf_raw = p_pwf_raw = p_twf_raw = 0.0
            
            if hasattr(model1, 'classes_'):
                classes = [str(c).strip().upper() for c in getattr(model1, 'classes_')]
                p_dict = dict(zip(classes, probs_list))
                p_normal = float(p_dict.get('NORMAL', p_dict.get('0', 0.0))) * 100.0
                p_hdf_raw = float(p_dict.get('HEAT DISSIPATION FAILURE (HDF)', p_dict.get('HDF', 0.0))) * 100.0
                p_osf_raw = float(p_dict.get('OVERSTRAIN FAILURE (OSF)', p_dict.get('OSF', 0.0))) * 100.0
                p_pwf_raw = float(p_dict.get('POWER FAILURE (PWF)', p_dict.get('PWF', 0.0))) * 100.0
                p_twf_raw = float(p_dict.get('TOOL WEAR FAILURE (TWF)', p_dict.get('TWF', 0.0))) * 100.0
                
                # Handling Binary Models effectively
                if '1' in p_dict or 'FAILURE' in p_dict:
                    raw_overall_risk = float(p_dict.get('1', p_dict.get('FAILURE', 0.0))) * 100.0
                else:
                    raw_overall_risk = 100.0 - p_normal
            else:
                p_normal = probs_list.__getitem__(0) * 100.0 if len(probs_list)>0 else 100.0
                raw_overall_risk = 100.0 - p_normal

            # --- Expert System Overrides ---
            expert_twf = min(50.0 + ((tool_wear - 200) / 40.0) * 45.0, 98.0) if tool_wear >= 200 else 0.0
            power = torque * rpm
            if power > 75000: expert_pwf = min(40.0 + ((power - 75000) / 15000.0) * 55.0, 98.0)
            elif power < 40000: expert_pwf = min(40.0 + ((40000 - power) / 10000.0) * 55.0, 98.0)
            else: expert_pwf = 0.0
            strain = torque * tool_wear
            strain_limit = 11000 if motor_options.get(motor_type) == 0 else (12000 if motor_options.get(motor_type) == 1 else 13000)
            warning_limit = strain_limit * 0.8
            expert_osf = min(40.0 + ((strain - warning_limit) / (strain_limit - warning_limit)) * 55.0, 98.0) if strain > warning_limit else 0.0
            temp_diff = process_temp - air_temp
            expert_hdf = min(40.0 + ((10.0 - temp_diff) / 2.0) * 55.0, 98.0) if temp_diff <= 10.0 and rpm < 1500 else 0.0

            p_hdf = max(p_hdf_raw, expert_hdf)
            p_osf = max(p_osf_raw, expert_osf)
            p_pwf = max(p_pwf_raw, expert_pwf)
            p_twf = max(p_twf_raw, expert_twf)

            calc_risk = (1.0 - ((1.0 - p_hdf/100.0) * (1.0 - p_osf/100.0) * (1.0 - p_pwf/100.0) * (1.0 - p_twf/100.0))) * 100.0
            overall_motor_risk = min(max(raw_overall_risk, calc_risk), 100.0)
            
            st.session_state.motor_risk = overall_motor_risk
            st.session_state.motor_faults = {'HDF': p_hdf, 'OSF': p_osf, 'PWF': p_pwf, 'TWF': p_twf}

            # Log Entry
            sys_status = "Safe" if overall_motor_risk < 40 else ("Warning" if overall_motor_risk < 75 else "Danger")
            highest_fault = max(st.session_state.motor_faults, key=st.session_state.motor_faults.get) if overall_motor_risk >= 40 else "NORMAL"
            status_color = "#00A65A" if sys_status=="Safe" else ("#F39C12" if sys_status=="Warning" else "#DD4B39")

            if analyze_btn:
                new_log = pd.DataFrame([{'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset': 'Motor Core', 'Risk Level': f"{overall_motor_risk:.1f}%", 'Primary FMEA': highest_fault, 'Status': sys_status}])
                st.session_state.inspection_log = pd.concat([new_log, st.session_state.inspection_log], ignore_index=True)

            # UI Rendering
            c_g1, c_g2 = st.columns((1, 2))
            with c_g1: st.plotly_chart(create_gauge(overall_motor_risk, "MOTOR RISK", status_color, h=280), use_container_width=True)
            with c_g2:
                df_f = pd.DataFrame({'Fault': ('Heat', 'Strain', 'Power', 'Wear'), 'Risk': (p_hdf, p_osf, p_pwf, p_twf)})
                fig_f = px.bar(df_f, x='Fault', y='Risk', color='Risk', color_continuous_scale='Reds', range_y=(0,100), title="Dynamic FMEA Distribution")
                fig_f.update_layout(height=280, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_f, use_container_width=True)

# --- TAB 2: BEARING VIBRATION ---
with tab2:
    cv1, cv2 = st.columns((1, 2.5))
    with cv1:
        st.markdown("#### 🎚️ NASA Run-to-Failure")
        sim_level = st.slider("Degradation Phase (%)", 0, 100, 10)
        r_val = 0.05 + (sim_level/100.0)**2 * 0.20
        k_val = 3.0 + (sim_level/100.0)**1.5 * 5.0
        sk_val = (sim_level/100.0) * 0.5
        p2p_val = r_val * 4.5
        cr_val = 3.5 + (sim_level/100.0)
        btn_vib = st.button("🔍 Analyze Vibration", use_container_width=True, type="primary")

    with cv2:
        if models_ready:
            bearing_df = pd.DataFrame(((r_val, k_val, sk_val, p2p_val, cr_val),), columns=('RMS', 'Kurtosis', 'Skewness', 'Peak_to_Peak', 'Crest_Factor'))
            scaled_b = scaler2.transform(bearing_df)
            probs_b = np.array(model2.predict_proba(scaled_b)).flatten().tolist()
            
            p_b_early = p_b_warn = p_b_fail = 0.0
            
            if hasattr(model2, 'classes_'):
                c_b = [str(c).strip().upper() for c in getattr(model2, 'classes_')]
                pb_dict = dict(zip(c_b, probs_b))
                if '1' in pb_dict and '2' not in pb_dict: # Binary handling
                    p_b_fail = float(pb_dict.get('1', pb_dict.get('FAILURE', 0.0))) * 100.0
                else:
                    p_b_early = float(pb_dict.get('1', 0.0)) * 100.0
                    p_b_warn = float(pb_dict.get('2', 0.0)) * 100.0
                    p_b_fail = float(pb_dict.get('3', 0.0)) * 100.0
            else:
                if len(probs_b) == 2: p_b_fail = probs_b.__getitem__(1) * 100.0
                else:
                    p_b_early = probs_b.__getitem__(1) * 100.0 if len(probs_b)>1 else 0.0
                    p_b_warn = probs_b.__getitem__(2) * 100.0 if len(probs_b)>2 else 0.0
                    p_b_fail = probs_b.__getitem__(3) * 100.0 if len(probs_b)>3 else 0.0
            
            b_risk = p_b_fail if (p_b_early==0 and p_b_warn==0) else ((p_b_early * 0.4) + (p_b_warn * 0.8) + (p_b_fail * 1.0))
            st.session_state.bearing_risk = b_risk
            
            b_color = "#00A65A" if b_risk < 40 else ("#F39C12" if b_risk < 75 else "#DD4B39")
            b_status = "Safe" if b_risk < 40 else ("Warning" if b_risk < 75 else "Danger")
            
            if btn_vib:
                new_log2 = pd.DataFrame([{'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset': 'Bearing', 'Risk Level': f"{b_risk:.1f}%", 'Primary FMEA': 'Vibration/Wear', 'Status': b_status}])
                st.session_state.inspection_log = pd.concat([new_log2, st.session_state.inspection_log], ignore_index=True)

            cb1, cb2 = st.columns(2)
            with cb1: st.plotly_chart(create_gauge(b_risk, "BEARING VIBRATION RISK", b_color, h=280), use_container_width=True)
            with cb2:
                df_b = pd.DataFrame({'Phase': ('Early Wear', 'Warning', 'Failure'), 'Prob': (p_b_early, p_b_warn, p_b_fail)})
                fig_b = px.bar(df_b, x='Prob', y='Phase', orientation='h', color='Prob', color_continuous_scale='Purples', range_x=(0,100), title="Degradation Phases")
                fig_b.update_layout(height=280, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_b, use_container_width=True)

# --- TAB 3: RELIABILITY FUSION ---
with tab3:
    m_frac = st.session_state.motor_risk / 100.0
    b_frac = st.session_state.bearing_risk / 100.0
    integrated_risk = (1.0 - ((1.0 - m_frac) * (1.0 - b_frac))) * 100.0
    f_color = "#00A65A" if integrated_risk < 40 else ("#F39C12" if integrated_risk < 75 else "#DD4B39")
    
    cf1, cf2, cf3 = st.columns((1, 0.2, 1))
    with cf1: st.markdown(f"<div class='metric-card' style='border-left-color:#3498DB;'><h4>Core Motor Risk</h4><h1 style='color:#3498DB;'>{st.session_state.motor_risk:.1f}%</h1></div>", unsafe_allow_html=True)
    with cf2: st.markdown("<h1 style='text-align:center; margin-top:30px; color:#aaa;'>✖</h1>", unsafe_allow_html=True)
    with cf3: st.markdown(f"<div class='metric-card' style='border-left-color:#9B59B6;'><h4>Bearing Vib Risk</h4><h1 style='color:#9B59B6;'>{st.session_state.bearing_risk:.1f}%</h1></div>", unsafe_allow_html=True)
    
    st.plotly_chart(create_gauge(integrated_risk, "OVERALL SYSTEM RISK (SERIES)", f_color, h=300), use_container_width=True)

# --- TAB 4: EXECUTIVE SUMMARY ---
with tab4:
    integrated_risk = (1.0 - ((1.0 - st.session_state.motor_risk/100.0) * (1.0 - st.session_state.bearing_risk/100.0))) * 100.0
    health_score = 100.0 - integrated_risk
    
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Asset Health Score", f"{health_score:.1f}%")
    e2.metric("Unplanned Downtime", f"{integrated_risk:.1f}%")
    e3.metric("Motor Stress", f"{st.session_state.motor_risk:.1f}%")
    e4.metric("Bearing Wear", f"{st.session_state.bearing_risk:.1f}%")
    
    er1, er2 = st.columns((1.5, 2))
    with er1:
        st.markdown("#### 🎯 Holistic FMEA Radar")
        cats = ('Heat (HDF)', 'Strain (OSF)', 'Power (PWF)', 'Wear (TWF)', 'Bearing (Vib)')
        vals = (st.session_state.motor_faults['HDF'], st.session_state.motor_faults['OSF'], st.session_state.motor_faults['PWF'], st.session_state.motor_faults['TWF'], st.session_state.bearing_risk)
        fig_r = go.Figure(data=go.Scatterpolar(r=vals, theta=cats, fill='toself', line_color='#DD4B39'))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=(0, 100))), showlegend=False, height=350, margin=dict(t=30, b=30))
        st.plotly_chart(fig_r, use_container_width=True)
    
    with er2:
        st.markdown("#### 📋 Automated FMEA Inspection Log")
        if not st.session_state.inspection_log.empty:
            st.dataframe(st.session_state.inspection_log.style.apply(highlight_fmea, axis=1), use_container_width=True, hide_index=True, height=350)
        else:
            st.info("Run diagnostics in Motor or Bearing tabs to populate the FMEA log.")

# =========================================
# 6. FOOTER
# =========================================
st.markdown("<br><hr><div style='text-align: center; color: #888; font-size: 11px;'><strong>© 2026 MotorGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>Confidential and Proprietary. DBA Research Project.</div>", unsafe_allow_html=True)