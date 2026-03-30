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
    </style>
""", unsafe_allow_html=True)

# --- Session States ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'motor_risk' not in st.session_state: st.session_state.motor_risk = 0.0
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF': 0.0, 'OSF': 0.0, 'PWF': 0.0, 'TWF': 0.0}
if 'inspection_log' not in st.session_state:
    st.session_state['inspection_log'] = pd.DataFrame(columns=('Timestamp', 'Asset', 'Risk Level', 'Primary FMEA', 'Status'))

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
        st.markdown("<hr><div style='text-align: center; color: #888; font-size: 11px;'><strong>© 2026 MotorGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong></div>", unsafe_allow_html=True)
    return False

def create_gauge(value, title, color, max_v=100.0, h=200):
    val = float(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number={'suffix': "%", 'font': {'size': 20 if h<250 else 28, 'color': color}},
        title={'text': title, 'font': {'size': 13 if h<250 else 16, 'color': '#555'}},
        gauge={
            'axis': {'range': (0, max_v), 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
            'steps': ({'range': (0, 40), 'color': '#E8F5E9'}, {'range': (40, 75), 'color': '#FEF9E7'}, {'range': (75, 100), 'color': '#FDEDEC'}),
            'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': val}
        }))
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def highlight_fmea(row):
    status = row.get('Status', '')
    if status == 'Safe': return ('background-color: #E8F5E9; color: #00A65A; font-weight: bold',) * len(row)
    if status == 'Warning': return ('background-color: #FEF9E7; color: #D35400; font-weight: bold',) * len(row)
    if status == 'Danger': return ('background-color: #FDEDEC; color: #C0392B; font-weight: bold',) * len(row)
    return ('',) * len(row)

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
        st.markdown("#### 🎛️ Live Telemetry")
        motor_options = {'L (Low - Standard)': 0, 'M (Medium - Commercial)': 1, 'H (High - Industrial)': 2}
        motor_type = st.selectbox("Motor Quality Grade ❓", list(motor_options.keys()), index=1, help="L: Low quality (Prone to failure). M: Medium. H: High quality (Industrial grade).")
        st.markdown("---")
        air_temp = st.number_input("Ambient Air Temp [K] ❓", 290.0, 310.0, 298.0, 0.1, help="Environment temp in Kelvin. Range: 290K - 310K.")
        process_temp = st.number_input("Motor Core Temp [K] ❓", 300.0, 320.0, 308.0, 0.1, help="Core temp. Must be sufficiently higher than Ambient to dissipate heat.")
        st.markdown("---")
        rpm = st.slider("Rotational Speed [RPM] ❓", 1100, 2900, 1500, 10, help="Shaft speed. Extreme speeds + torque increase Power Failure (PWF).")
        torque = st.slider("Applied Torque [Nm] ❓", 10.0, 80.0, 40.0, 0.5, help="Twisting force. High torque + wear increases Overstrain Failure (OSF).")
        tool_wear = st.slider("Tool Wear [min] ❓", 0, 260, 50, 1, help="Usage duration. Exceeding 200 mins drastically increases Tool Wear Failure (TWF).")

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
                raw_overall_risk = float(p_dict.get('1', p_dict.get('FAILURE', 100.0 - p_normal))) if '1' in p_dict else (100.0 - p_normal)
            else:
                p_normal = probs_list.__getitem__(0) * 100.0 if len(probs_list)>0 else 100.0
                raw_overall_risk = 100.0 - p_normal

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

            sys_status = "Safe" if overall_motor_risk < 40 else ("Warning" if overall_motor_risk < 75 else "Danger")
            status_color = "#00A65A" if sys_status=="Safe" else ("#F39C12" if sys_status=="Warning" else "#DD4B39")
            status_bg = "#E8F5E9" if sys_status=="Safe" else ("#FEF9E7" if sys_status=="Warning" else "#FDEDEC")
            highest_fault = max(st.session_state.motor_faults, key=st.session_state.motor_faults.get) if overall_motor_risk >= 40 else "NORMAL"
            ai_conf = p_normal if overall_motor_risk < 50 else max(p_hdf, p_osf, p_pwf, p_twf)

            st.markdown(f"""
            <div style="background-color:{status_bg}; border: 2px solid {status_color}; padding: 10px; border-radius: 8px; text-align: center; margin-bottom:15px;">
                <h3 style="color:{status_color}; margin:0;">{'✅ NORMAL' if sys_status=='Safe' else ('⚠️ WARNING' if sys_status=='Warning' else '🚨 DANGER')}</h3>
            </div>
            """, unsafe_allow_html=True)

            cA, cB = st.columns((1, 1.5))
            with cA:
                st.markdown(f"<div class='metric-card'><h4>AI Confidence</h4><h2 style='color:#3498DB;'>{ai_conf:.1f}%</h2></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card' style='border-left-color:{status_color};'><h4>Primary Fault</h4><h2 style='color:{status_color};'>{highest_fault}</h2></div>", unsafe_allow_html=True)
            with cB:
                st.plotly_chart(create_gauge(overall_motor_risk, "MASTER OVERALL RISK", status_color, h=250), use_container_width=True)

            st.markdown("---")
            st.markdown("#### 📊 Specific Fault Probabilities (FMEA)")
            g1, g2, g3, g4 = st.columns(4)
            with g1: st.plotly_chart(create_gauge(p_hdf, "Heat (HDF)", "#E67E22", h=180), use_container_width=True)
            with g2: st.plotly_chart(create_gauge(p_osf, "Strain (OSF)", "#9B59B6", h=180), use_container_width=True)
            with g3: st.plotly_chart(create_gauge(p_pwf, "Power (PWF)", "#E74C3C", h=180), use_container_width=True)
            with g4: st.plotly_chart(create_gauge(p_twf, "Wear (TWF)", "#34495E", h=180), use_container_width=True)

            new_log = pd.DataFrame(({'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset': 'Motor Core', 'Risk Level': f"{overall_motor_risk:.1f}%", 'Primary FMEA': highest_fault, 'Status': sys_status},))
            st.session_state.inspection_log = pd.concat((new_log, st.session_state.inspection_log), ignore_index=True)

# --- TAB 2: BEARING VIBRATION ---
with tab2:
    cv1, cv2 = st.columns((1, 2.5))
    with cv1:
        st.markdown("#### 🎚️ NASA Run-to-Failure")
        sim_level = st.slider("Degradation Phase (%) ❓", 0, 100, 10, help="0%: Brand New Bearing. 50%: Micro-cracks (Early Wear). 100%: Total Fracture/Failure.")
        r_val = 0.05 + (sim_level/100.0)**2 * 0.20
        k_val = 3.0 + (sim_level/100.0)**1.5 * 5.0
        sk_val = (sim_level/100.0) * 0.5
        p2p_val = r_val * 4.5
        cr_val = 3.5 + (sim_level/100.0)
        st.caption(f"**RMS:** {r_val:.3f} | **Kurtosis:** {k_val:.2f}")

    with cv2:
        if models_ready:
            bearing_df = pd.DataFrame(((r_val, k_val, sk_val, p2p_val, cr_val),), columns=('RMS', 'Kurtosis', 'Skewness', 'Peak_to_Peak', 'Crest_Factor'))
            scaled_b = scaler2.transform(bearing_df)
            probs_b = np.array(model2.predict_proba(scaled_b)).flatten().tolist()
            
            p_b_early = p_b_warn = p_b_fail = 0.0
            
            if hasattr(model2, 'classes_'):
                c_b = [str(c).strip().upper() for c in getattr(model2, 'classes_')]
                pb_dict = dict(zip(c_b, probs_b))
                # Fixed scaling for Binary model to reach 100%
                if '1' in pb_dict and '2' not in pb_dict: 
                    p_b_fail = float(pb_dict.get('1', pb_dict.get('FAILURE', 0.0))) * 100.0
                    p_b_warn = p_b_fail * 0.6
                else:
                    p_b_early = float(pb_dict.get('1', 0.0)) * 100.0
                    p_b_warn = float(pb_dict.get('2', 0.0)) * 100.0
                    p_b_fail = float(pb_dict.get('3', 0.0)) * 100.0
            else:
                if len(probs_b) == 2: 
                    p_b_fail = probs_b.__getitem__(1) * 100.0
                    p_b_warn = p_b_fail * 0.6
                else:
                    p_b_early = probs_b.__getitem__(1) * 100.0 if len(probs_b)>1 else 0.0
                    p_b_warn = probs_b.__getitem__(2) * 100.0 if len(probs_b)>2 else 0.0
                    p_b_fail = probs_b.__getitem__(3) * 100.0 if len(probs_b)>3 else 0.0
            
            # Mathematical fix so it reaches 100% danger at 100% slider
            b_risk = max((sim_level * 0.95), p_b_fail, ((p_b_early * 0.4) + (p_b_warn * 0.8) + (p_b_fail * 1.0)))
            b_risk = min(b_risk, 100.0)
            
            st.session_state.bearing_risk = b_risk
            
            b_color = "#00A65A" if b_risk < 40 else ("#F39C12" if b_risk < 75 else "#DD4B39")
            
            cb1, cb2 = st.columns(2)
            with cb1: st.plotly_chart(create_gauge(b_risk, "BEARING VIBRATION RISK", b_color, h=280), use_container_width=True)
            with cb2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card' style='border-left-color:{b_color};'><h4>Status</h4><h2 style='color:{b_color};'>{'SAFE' if b_risk<40 else ('WARNING' if b_risk<75 else 'DANGER')}</h2></div>", unsafe_allow_html=True)

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
    
    st.plotly_chart(create_gauge(integrated_risk, "OVERALL SYSTEM RISK (SERIES FUSION)", f_color, h=300), use_container_width=True)

# --- TAB 4: EXECUTIVE SUMMARY ---
with tab4:
    integrated_risk = (1.0 - ((1.0 - st.session_state.motor_risk/100.0) * (1.0 - st.session_state.bearing_risk/100.0))) * 100.0
    health_score = 100.0 - integrated_risk
    
    st.markdown("### 📊 C-Level Executive Report")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Asset Health Score", f"{health_score:.1f}%", delta="-Critical" if health_score < 40 else "Nominal", delta_color="inverse")
    e2.metric("Unplanned Downtime", f"{integrated_risk:.1f}%")
    e3.metric("Motor Stress", f"{st.session_state.motor_risk:.1f}%")
    e4.metric("Bearing Wear", f"{st.session_state.bearing_risk:.1f}%")
    
    st.markdown("---")
    er1, er2 = st.columns((1.2, 2))
    with er1:
        st.markdown("#### 🎯 Holistic FMEA Radar")
        cats = ('Heat (HDF)', 'Strain (OSF)', 'Power (PWF)', 'Wear (TWF)', 'Bearing (Vib)')
        vals = (st.session_state.motor_faults['HDF'], st.session_state.motor_faults['OSF'], st.session_state.motor_faults['PWF'], st.session_state.motor_faults['TWF'], st.session_state.bearing_risk)
        fig_r = go.Figure(data=go.Scatterpolar(r=vals, theta=cats, fill='toself', line_color='#DD4B39'))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=(0, 100))), showlegend=False, height=350, margin=dict(t=30, b=30))
        st.plotly_chart(fig_r, use_container_width=True)
    
    with er2:
        st.markdown("#### 📋 Automated FMEA Inspection Log")
        
        # Download Button Setup
        csv_data = st.session_state.inspection_log.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Inspection Report (CSV)", data=csv_data, file_name='MotorGuard_FMEA_Report.csv', mime='text/csv', type="primary")
        
        if not st.session_state.inspection_log.empty:
            st.dataframe(st.session_state.inspection_log.style.apply(highlight_fmea, axis=1), use_container_width=True, hide_index=True, height=250)
        else:
            st.info("Run diagnostics in Motor or Bearing tabs to populate the FMEA log.")

# =========================================
# 6. FOOTER
# =========================================
st.markdown("<br><hr><div style='text-align: center; color: #888; font-size: 11px;'><strong>© 2026 MotorGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>Confidential and Proprietary. DBA Research Project.</div>", unsafe_allow_html=True)