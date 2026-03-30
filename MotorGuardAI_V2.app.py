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
# 1. PAGE SETUP & GLOBAL STYLES
# =========================================
st.set_page_config(page_title="MotorGuard AI Pro", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

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
    except Exception as e:
        return None, None, None, None, False

scaler1, model1, scaler2, model2, models_ready = load_ai_assets()

# =========================================
# 3. AUTHENTICATION & GAUGES
# =========================================
def check_password():
    if st.session_state.authenticated: return True
    col1, col2, col3 = st.columns(list((1.5, 2.0, 1.5))) 
    with col2: 
        st.markdown("""
        <div style="background-color:#F8F9FA; padding:30px; border-radius:15px; border-top: 5px solid #DD4B39; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 50px;">
            <h2 style="color:#1E1E1E; margin-bottom: 5px;">MotorGuard <span style="color:#DD4B39;">Enterprise</span></h2>
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

st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin:0;">⚙️ MotorGuard AI <span style="color:#DD4B39;">Enterprise</span></h2>
            <p style="margin:0; font-size: 14px; color:#CCCCCC;">Advanced Asset Performance Management | DBA Research</p>
        </div>
        <div style="text-align: right;"><h4 style="margin:0; color:#00ff88;">⚡ LIVE TELEMETRY</h4></div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(list(("⚙️ Motor Diagnostics", "📳 Bearing Vibration", "🔗 Reliability Fusion", "📊 Executive Summary")))

# ==============================================================================
# TAB 1: MOTOR DIAGNOSTICS
# ==============================================================================
with tab1:
    c_in, c_out = st.columns(list((1, 2.5)))
    with c_in:
        st.markdown("#### 🎛️ Live Sensor Telemetry")
        motor_options = {
            'L (Low - Standard Build)': 0,
            'M (Medium - Commercial)': 1,
            'H (High - Industrial Grade)': 2
        }
        motor_type = st.selectbox(
            "Motor Quality Grade ❓", 
            list(motor_options.keys()), 
            index=1,
            help="L: Low quality (More prone to failure and lower limits). M: Medium quality. H: High quality (Durable industrial grade with high tolerance)."
        )
        
        st.markdown("---")
        st.markdown("#### 🌡️ Thermal Metrics")
        air_temp = st.number_input("Ambient Air Temp [K] ❓", min_value=290.0, max_value=310.0, value=298.0, step=0.1, help="Temperature of the surrounding environment in Kelvin. Range: 290K - 310K.")
        process_temp = st.number_input("Motor Core Temp [K] ❓", min_value=300.0, max_value=320.0, value=308.0, step=0.1, help="Operating temperature of the motor core. Range: 300K - 320K. (Needs to be sufficiently higher than Ambient Temp for heat dissipation).")
        
        st.markdown("---")
        st.markdown("#### ⚙️ Mechanical Metrics")
        rpm = st.slider("Rotational Speed [RPM] ❓", min_value=1100, max_value=2900, value=1500, step=10, help="Motor shaft speed. Extreme high speeds combined with torque increase Power Failure (PWF) risk.")
        torque = st.slider("Applied Torque [Nm] ❓", min_value=10.0, max_value=80.0, value=40.0, step=0.5, help="Twisting force applied by the motor. High torque combined with tool wear increases Overstrain Failure (OSF) risk.")
        tool_wear = st.slider("Tool Wear Level [min] ❓", min_value=0, max_value=260, value=50, step=1, help="Usage duration of the attached tool. Exceeding 200 mins drastically increases Tool Wear Failure (TWF) risk.")

    with c_out:
        if models_ready:
            c_air = 'Air temperature ' + chr(91) + 'K' + chr(93)
            c_proc = 'Process temperature ' + chr(91) + 'K' + chr(93)
            c_rpm = 'Rotational speed ' + chr(91) + 'rpm' + chr(93)
            c_trq = 'Torque ' + chr(91) + 'Nm' + chr(93)
            c_wear = 'Tool wear ' + chr(91) + 'min' + chr(93)
            
            raw_df = pd.DataFrame({
                'Type': list((motor_options[motor_type],)),
                c_air: list((air_temp,)),
                c_proc: list((process_temp,)),
                c_rpm: list((rpm,)),
                c_trq: list((torque,)),
                c_wear: list((tool_wear,))
            })
            
            cols_to_scale = list((c_air, c_proc, c_rpm, c_trq, c_wear))
            processed_df = raw_df.copy()
            processed_df[cols_to_scale] = scaler1.transform(raw_df[cols_to_scale])
            
            # --- AI PREDICTION ---
            prediction_matrix = getattr(model1, 'predict')(processed_df)
            diag_label = str(np.array(prediction_matrix).flatten().item(0)).strip()
            
            probs_matrix = getattr(model1, 'predict_proba')(processed_df)
            probs_list = np.array(probs_matrix).flatten().tolist()
            
            if hasattr(model1, 'classes_'):
                classes = list((str(c).strip().upper() for c in getattr(model1, 'classes_')))
                probs_dict = dict(zip(classes, probs_list))
                
                p_normal = float(probs_dict.get('NORMAL', 0.0)) * 100.0
                
                p_hdf_raw = float(probs_dict.get('HEAT DISSIPATION FAILURE (HDF)', probs_dict.get('HDF', 0.0))) * 100.0
                p_osf_raw = float(probs_dict.get('OVERSTRAIN FAILURE (OSF)', probs_dict.get('OSF', 0.0))) * 100.0
                p_pwf_raw = float(probs_dict.get('POWER FAILURE (PWF)', probs_dict.get('PWF', 0.0))) * 100.0
                p_twf_raw = float(probs_dict.get('TOOL WEAR FAILURE (TWF)', probs_dict.get('TWF', 0.0))) * 100.0
                
                raw_overall_risk = 100.0 - p_normal
                
                if raw_overall_risk >= 85.0:
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
            else:
                p_normal = float(probs_list.pop(0)) * 100.0 if len(probs_list) > 0 else 0.0
                p_hdf = p_osf = p_pwf = p_twf = 0.0
                overall_risk_pct = 100.0 - p_normal

            st.session_state.motor_risk = overall_risk_pct
            st.session_state.motor_faults = {'HDF': p_hdf, 'OSF': p_osf, 'PWF': p_pwf, 'TWF': p_twf}

            if overall_risk_pct < 40.0:
                status_color = "#00A65A" 
                status_bg = "#E8F5E9"
                status_title = "✅ SYSTEM NORMAL"
                sys_status = "Safe"
                diag_display = "NORMAL"
            elif overall_risk_pct < 75.0:
                status_color = "#F39C12" 
                status_bg = "#FEF9E7"
                status_title = f"⚠️ ELEVATED RISK"
                sys_status = "Warning"
                fault_dict = {'Heat (HDF)': p_hdf, 'Overstrain (OSF)': p_osf, 'Power (PWF)': p_pwf, 'Wear (TWF)': p_twf}
                diag_display = max(fault_dict, key=fault_dict.get)
            else:
                status_color = "#DD4B39" 
                status_bg = "#FDEDEC"
                status_title = f"🚨 DANGER: {diag_label}"
                sys_status = "Danger"
                diag_display = diag_label
                
            ai_conf = p_normal if overall_risk_pct < 50 else overall_risk_pct

            cA, cB = st.columns(list((1, 1.5)))
            with cA:
                st.markdown(f"""
                <div style="background-color:{status_bg}; border: 2px solid {status_color}; padding: 15px; border-radius: 8px; text-align: center; margin-bottom:15px;">
                    <h3 style="color:{status_color}; margin:0;">{status_title}</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><h4>AI Confidence Score</h4><h2 style='color:#3498DB;'>{ai_conf:.1f}%</h2></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card' style='border-left-color:{status_color};'><h4>Primary Diagnosis</h4><h2 style='color:{status_color};'>{diag_display}</h2></div>", unsafe_allow_html=True)
            with cB:
                st.markdown("""
                <div class="zone-bar">
                    <div class="zone-safe">SAFE (0-40)</div>
                    <div class="zone-warn">WARN (40-75)</div>
                    <div class="zone-dang">DANGER (75-100)</div>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(create_gauge(overall_risk_pct, "MASTER OVERALL RISK", status_color, h=280), use_container_width=True)

            st.markdown("---")
            st.markdown("#### 📊 Specific FMEA Fault Probabilities")
            g1, g2, g3, g4 = st.columns(4)
            with g1: st.plotly_chart(create_gauge(p_hdf, "Heat (HDF)", "#E67E22", h=160), use_container_width=True)
            with g2: st.plotly_chart(create_gauge(p_osf, "Strain (OSF)", "#9B59B6", h=160), use_container_width=True)
            with g3: st.plotly_chart(create_gauge(p_pwf, "Power (PWF)", "#E74C3C", h=160), use_container_width=True)
            with g4: st.plotly_chart(create_gauge(p_twf, "Wear (TWF)", "#34495E", h=160), use_container_width=True)

            # Auto-update Log for Motor
            motor_short_name = str(motor_type).split(' ').pop(0)
            new_log_raw = pd.DataFrame(list(({'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset': 'Motor Core', 'Risk Level': f"{overall_risk_pct:.1f}%", 'Primary FMEA': diag_display, 'Status': sys_status, 'Grade': motor_short_name, 'Air Temp [K]': air_temp, 'Core Temp [K]': process_temp, 'RPM': rpm, 'Torque [Nm]': torque, 'Wear [min]': tool_wear, 'Vib RMS': '-', 'Vib Kurt': '-'},)))
            # Keep only last 30 logs to avoid memory issues
            st.session_state.inspection_log = pd.concat(list((new_log_raw, st.session_state.inspection_log)), ignore_index=True).head(30)
        else:
            st.info("⚠️ Waiting for AI Models to load.")

# ==============================================================================
# TAB 2: BEARING VIBRATION
# ==============================================================================
with tab2:
    cv1, cv2 = st.columns(list((1.5, 1)))
    with cv1:
        st.markdown("#### 🎚️ Simulator & Key Metrics")
        sim_level = st.slider("Degradation Phase (%) ❓", 0, 100, 10, help="Simulates NASA Run-to-Failure test. 0%: Brand New. 50%: Micro-cracks. 100%: Total Fracture.")
        
        r_val = 0.05 + (sim_level/100.0)**2 * 0.20
        k_val = 3.0 + (sim_level/100.0)**1.5 * 5.0
        sk_val = (sim_level/100.0) * 0.5
        p2p_val = r_val * 4.5
        cr_val = 3.5 + (sim_level/100.0)
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("RMS ❓", f"{r_val:.3f}", help="Root Mean Square: Overall vibration energy.")
        k2.metric("Kurtosis ❓", f"{k_val:.2f}", help="Peakedness of signal. Spikes indicate impacts.")
        k3.metric("Peak-to-Peak ❓", f"{p2p_val:.2f}", help="Max amplitude difference.")
        k4.metric("Crest Factor ❓", f"{cr_val:.2f}", help="Ratio of peak to RMS.")

    with cv2:
        p_b_fail = (sim_level / 100.0)**2 * 100.0
        p_b_warn = (sim_level / 100.0) * (1.0 - (sim_level / 100.0)) * 200.0
        p_b_early = (1.0 - (sim_level / 100.0))**2 * 100.0 if sim_level > 20 else (sim_level * 2.0)
        p_b_norm = max(0.0, 100.0 - (p_b_fail + p_b_warn + p_b_early))
        
        tot = p_b_fail + p_b_warn + p_b_early + p_b_norm
        if tot > 0:
            p_b_fail, p_b_warn, p_b_early, p_b_norm = (p_b_fail/tot)*100, (p_b_warn/tot)*100, (p_b_early/tot)*100, (p_b_norm/tot)*100
        
        raw_b_risk = (p_b_early * 0.3) + (p_b_warn * 0.7) + (p_b_fail * 1.0)
        
        b_risk = 100.0 if raw_b_risk >= 85.0 else raw_b_risk
        st.session_state.bearing_risk = b_risk
        
        b_color = "#00A65A" if b_risk < 40 else ("#F39C12" if b_risk < 75 else "#DD4B39")
        b_status = "Safe" if b_risk < 40 else ("Warning" if b_risk < 75 else "Danger")
        st.plotly_chart(create_gauge(b_risk, "BEARING VIBRATION RISK", b_color, max_v=100.0, h=280), use_container_width=True)

    st.markdown("---")
    df_b = pd.DataFrame({'Phase': list(('Normal', 'Early Wear', 'Warning', 'Failure')), 'Probability': list((p_b_norm, p_b_early, p_b_warn, p_b_fail))})
    color_map = {'Normal': '#2ecc71', 'Early Wear': '#f1c40f', 'Warning': '#e67e22', 'Failure': '#e74c3c'}
    fig_b = px.bar(df_b, x='Probability', y='Phase', orientation='h', color='Phase', color_discrete_map=color_map, title="AI Phase Classification")
    fig_b.update_layout(height=250)
    st.plotly_chart(fig_b, use_container_width=True)
    
    # Auto-update Log for Bearing
    new_log_b = pd.DataFrame(list(({'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset': 'Bearing', 'Risk Level': f"{b_risk:.1f}%", 'Primary FMEA': 'Vibration Anomalies', 'Status': b_status, 'Grade': '-', 'Air Temp [K]': '-', 'Core Temp [K]': '-', 'RPM': '-', 'Torque [Nm]': '-', 'Wear [min]': '-', 'Vib RMS': round(r_val, 3), 'Vib Kurt': round(k_val, 2)},)))
    st.session_state.inspection_log = pd.concat(list((new_log_b, st.session_state.inspection_log)), ignore_index=True).head(30)

# ==============================================================================
# TAB 3: RELIABILITY FUSION
# ==============================================================================
with tab3:
    st.markdown("### 🔗 Master Reliability Fusion (Series System)")
    m_frac = st.session_state.motor_risk / 100.0
    b_frac = st.session_state.bearing_risk / 100.0
    integrated_risk = (1.0 - ((1.0 - m_frac) * (1.0 - b_frac))) * 100.0
    f_color = "#00A65A" if integrated_risk < 40 else ("#F39C12" if integrated_risk < 75 else "#DD4B39")
    
    fc1, fc2, fc3 = st.columns(3)
    fc1.markdown(f"<div class='metric-card' style='border-left-color:#3498DB;'><h4>Motor Risk Factor</h4><h2 style='color:#3498DB;'>{st.session_state.motor_risk:.1f}%</h2></div>", unsafe_allow_html=True)
    fc2.markdown(f"<div class='metric-card' style='border-left-color:#9B59B6;'><h4>Bearing Risk Factor</h4><h2 style='color:#9B59B6;'>{st.session_state.bearing_risk:.1f}%</h2></div>", unsafe_allow_html=True)
    fc3.markdown(f"<div class='metric-card' style='border-left-color:{f_color};'><h4>Total Integrated Risk</h4><h2 style='color:{f_color};'>{integrated_risk:.1f}%</h2></div>", unsafe_allow_html=True)
    
    col_w, col_g = st.columns(list((1.5, 1)))
    with col_w:
        st.markdown("#### 📊 Risk Contributor Analysis")
        fig_comp = go.Figure(data=list((
            go.Bar(name='Motor Factor', x=list(('Risk Component',)), y=list((st.session_state.motor_risk,)), marker_color='#3498DB', text=list((f"{st.session_state.motor_risk:.1f}%",)), textposition='auto'),
            go.Bar(name='System (Series)', x=list(('Risk Component',)), y=list((integrated_risk,)), marker_color=f_color, text=list((f"{integrated_risk:.1f}%",)), textposition='auto'),
            go.Bar(name='Bearing Factor', x=list(('Risk Component',)), y=list((st.session_state.bearing_risk,)), marker_color='#9B59B6', text=list((f"{st.session_state.bearing_risk:.1f}%",)), textposition='auto')
        )))
        fig_comp.update_layout(barmode='group', height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_g:
        st.plotly_chart(create_gauge(integrated_risk, "OVERALL SYSTEM RISK", f_color, max_v=100.0, h=300), use_container_width=True)
        action_text = "EMERGENCY SHUTDOWN REQUIRED" if integrated_risk > 75 else ("SCHEDULE PREVENTIVE MAINTENANCE" if integrated_risk > 40 else "OPTIMAL PRODUCTION CONTINUES")
        st.markdown(f"<div style='text-align:center; padding:15px; background-color:#F8F9FA; border-radius:5px; border: 2px dashed {f_color};'><h4 style='margin:0; color:#555;'>AI Recommended Action:</h4><h3 style='margin:0; color:{f_color};'>{action_text}</h3></div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 4: EXECUTIVE SUMMARY
# ==============================================================================
with tab4:
    integrated_risk = (1.0 - ((1.0 - st.session_state.motor_risk/100.0) * (1.0 - st.session_state.bearing_risk/100.0))) * 100.0
    health_score = 100.0 - integrated_risk
    
    rul_days = int((health_score / 100.0) * 365 * 3) 
    rul_text = f"{rul_days} Days" if rul_days > 30 else "IMMINENT FAILURE"
    
    st.markdown("### 📊 Executive Summary & Reporting")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Asset Health Score", f"{health_score:.1f}%", help="Health = 100% - Total Integrated Risk.")
    e2.metric("Unplanned Downtime Prob", f"{integrated_risk:.1f}%", help="Probability of unexpected machine failure.")
    e3.metric("Est. Remaining Useful Life (RUL)", rul_text, help="Estimated days until critical failure if current stress continues.")
    e4.metric("Maintenance Urgency", "CRITICAL" if integrated_risk > 75 else ("MODERATE" if integrated_risk > 40 else "LOW"))
    
    st.markdown("---")
    er1, er2 = st.columns(list((1.5, 2)))
    with er1:
        st.markdown("#### 🎯 Holistic Fault Radar")
        cats = list(('Heat (HDF)', 'Strain (OSF)', 'Power (PWF)', 'Wear (TWF)', 'Bearing (Vib)'))
        vals = list((st.session_state.motor_faults.get('HDF',0), st.session_state.motor_faults.get('OSF',0), st.session_state.motor_faults.get('PWF',0), st.session_state.motor_faults.get('TWF',0), st.session_state.bearing_risk))
        fig_r = go.Figure(data=go.Scatterpolar(r=vals, theta=cats, fill='toself', line_color='#DD4B39'))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=list((0, 100)))), showlegend=False, height=350, margin=dict(t=30, b=30))
        st.plotly_chart(fig_r, use_container_width=True)
    with er2:
        st.markdown("#### 🛡️ System Viability Index (Safe vs At Risk)")
        df_viability = pd.DataFrame({
            'Component': list(('Motor Core', 'Bearing Assembly', 'Total System')),
            'Safe %': list((100.0 - st.session_state.motor_risk, 100.0 - st.session_state.bearing_risk, health_score)),
            'At Risk %': list((st.session_state.motor_risk, st.session_state.bearing_risk, integrated_risk))
        })
        fig_viab = px.bar(
            df_viability, 
            x="Component", 
            y=list(("Safe %", "At Risk %")), 
            title="Operational Viability Analysis",
            color_discrete_sequence=list(('#2ecc71', '#e74c3c'))
        )
        fig_viab.update_layout(height=350, barmode='stack', margin=dict(t=40, b=10))
        st.plotly_chart(fig_viab, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 📋 Live FMEA Diagnostic Log")
    if not st.session_state.inspection_log.empty:
        csv_data = st.session_state.inspection_log.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Full Executive Report (CSV)", data=csv_data, file_name='MotorGuard_Executive_Report.csv', mime='text/csv', type="primary")
        st.dataframe(st.session_state.inspection_log.style.apply(highlight_fmea, axis=1), use_container_width=True, hide_index=True, height=250)

# =========================================
# 6. FOOTER
# =========================================
st.markdown("<br><hr><div style='text-align: center; color: #888; font-size: 11px;'><strong>© 2026 MotorGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>Confidential and Proprietary. DBA Research Project.</div>", unsafe_allow_html=True)