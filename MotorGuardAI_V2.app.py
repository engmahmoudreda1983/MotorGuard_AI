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
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c5364 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================
# 2. AI MODELS LOADING & MOCK FALLBACK
# =========================================
# Dummy models in case .pkl files are not in the same folder during demo
class DummyMotorModel:
    def predict_proba(self, X): return [[0.9, 0.02, 0.03, 0.03, 0.02]]
    @property
    def classes_(self): return ['NORMAL', 'HDF', 'OSF', 'PWF', 'TWF']

class DummyBearingModel:
    def predict_proba(self, X): return [[0.8, 0.15, 0.04, 0.01]]
    @property
    def classes_(self): return ['0: Normal', '1: Early Wear', '2: Warning', '3: Failure']

class DummyScaler:
    def transform(self, X): return X

@st.cache_resource
def load_ai_assets():
    try:
        scaler1 = joblib.load('scaler.pkl')
        model1 = joblib.load('motorguard_model.pkl')
        scaler2 = joblib.load('nasa_bearing_scaler.pkl')
        model2 = joblib.load('nasa_bearing_model.pkl')
        return scaler1, model1, scaler2, model2, True
    except Exception as e:
        return DummyScaler(), DummyMotorModel(), DummyScaler(), DummyBearingModel(), False

scaler1, model1, scaler2, model2, models_loaded = load_ai_assets()

# =========================================
# 3. HELPER FUNCTIONS
# =========================================
def create_industrial_gauge(value, title, color_theme, max_val=100, height=220):
    safe_value = min(max(float(value), 0.0), max_val)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=safe_value,
        number={'suffix': "%", 'font': {'size': 24, 'color': color_theme, 'family': 'Arial Black'}},
        title={'text': title, 'font': {'size': 14, 'color': '#555', 'weight': 'bold'}},
        gauge={
            'axis': {'range': (0, max_val), 'tickwidth': 1, 'tickcolor': "darkblue", 'tickvals': (0, 40, 75, 100)},
            'bar': {'color': color_theme},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': (0, 40), 'color': '#E8F5E9'},
                {'range': (40, 75), 'color': '#FFF9C4'},
                {'range': (75, 100), 'color': '#FFCDD2'}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': safe_value}
        }))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# =========================================
# 4. MAIN APP HEADER
# =========================================
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin:0; font-family: 'Arial Black', sans-serif;">🏭 MotorGuard APM <span style="color:#00ff88;">Enterprise</span></h1>
            <p style="margin:0; font-size: 16px; color:#e0e0e0;">Unified Asset Performance Management & Deep Diagnostics</p>
        </div>
        <div style="text-align: right;">
            <p style="margin:0; color:#00ff88; font-weight: bold;">Status: System Online</p>
            <p style="margin:0; font-size: 12px; color:#ccc;">Developed by Eng. Mahmoud Reda Ibrahim</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.warning("⚠️ لم يتم العثور على ملفات النماذج الحقيقية (.pkl). النظام يعمل الآن على وضع المحاكاة (Simulation Mode) لأغراض العرض المبدئي.")

# =========================================
# 5. TABS SETUP
# =========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Model 1: CNC Motor Diagnostics", 
    "📳 Model 2: Vibration Analysis (NASA)", 
    "🔗 System Fusion & FMEA", 
    "📊 Executive Command Center"
])

# --- STATE VARIABLES ---
if 'motor_risk' not in st.session_state: st.session_state.motor_risk = 0.0
if 'motor_faults' not in st.session_state: st.session_state.motor_faults = {'HDF':0, 'OSF':0, 'PWF':0, 'TWF':0}
if 'bearing_risk' not in st.session_state: st.session_state.bearing_risk = 0.0
if 'bearing_status' not in st.session_state: st.session_state.bearing_status = "Normal"

# =========================================
# TAB 1: CNC MOTOR (MODEL 1)
# =========================================
with tab1:
    st.markdown("### ⚙️ Thermomechanical AI Engine (AI4I Dataset)")
    
    col_in, col_out = st.columns()
    
    with col_in:
        st.markdown("<div style='background-color:#f4f6f9; padding:15px; border-radius:10px;'>", unsafe_allow_html=True)
        motor_type = st.selectbox("Motor Grade", ['L (Low)', 'M (Medium)', 'H (High)'], index=1)
        air_temp = st.number_input("Ambient Temp [K]", 290.0, 310.0, 298.0)
        process_temp = st.number_input("Core Temp [K]", 300.0, 320.0, 308.0)
        rpm = st.slider("Speed [RPM]", 1100, 2900, 1500)
        torque = st.slider("Torque [Nm]", 10.0, 80.0, 40.0)
        tool_wear = st.slider("Tool Wear [min]", 0, 260, 50)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_out:
        # Data Prep & Prediction
        type_val = 0 if 'L' in motor_type else (1 if 'M' in motor_type else 2)
        raw_df1 = pd.DataFrame({'Type': [type_val], 'Air temperature [K]': [air_temp], 'Process temperature [K]': [process_temp], 'Rotational speed [rpm]': [rpm], 'Torque [Nm]': [torque], 'Tool wear [min]': [tool_wear]})
        processed_df1 = raw_df1.copy()
        cols_to_scale = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        processed_df1[cols_to_scale] = scaler1.transform(raw_df1[cols_to_scale])
        
        probs1 = np.array(model1.predict_proba(processed_df1)).flatten()
        if hasattr(model1, 'classes_'):
            p_dict = dict(zip([str(c).upper() for c in model1.classes_], probs1))
            p_norm = p_dict.get('NORMAL', 0)*100
            p_hdf = p_dict.get('HEAT DISSIPATION FAILURE (HDF)', p_dict.get('HDF',0))*100
            p_osf = p_dict.get('OVERSTRAIN FAILURE (OSF)', p_dict.get('OSF',0))*100
            p_pwf = p_dict.get('POWER FAILURE (PWF)', p_dict.get('PWF',0))*100
            p_twf = p_dict.get('TOOL WEAR FAILURE (TWF)', p_dict.get('TWF',0))*100
        else:
            p_norm = probs1*100
            p_hdf = p_osf = p_pwf = p_twf = 0
            
        # Cumulative Logic for Motor
        p_fail_hdf = p_hdf/100; p_fail_osf = p_osf/100; p_fail_pwf = p_pwf/100; p_fail_twf = p_twf/100
        motor_overall_risk = (1.0 - ((1-p_fail_hdf)*(1-p_fail_osf)*(1-p_fail_pwf)*(1-p_fail_twf))) * 100
        
        # Save to Session
        st.session_state.motor_risk = motor_overall_risk
        st.session_state.motor_faults = {'HDF':p_hdf, 'OSF':p_osf, 'PWF':p_pwf, 'TWF':p_twf}
        
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(create_industrial_gauge(motor_overall_risk, "Motor Core Risk", "#3498DB", height=250), use_container_width=True)
        with c2:
            st.markdown("#### Fault Distribution")
            df_faults = pd.DataFrame({'Fault': ['Heat', 'Strain', 'Power', 'Wear'], 'Risk': [p_hdf, p_osf, p_pwf, p_twf]})
            fig_bar = px.bar(df_faults, x='Fault', y='Risk', color='Risk', color_continuous_scale='Reds', range_y=)
            fig_bar.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

# =========================================
# TAB 2: VIBRATION (MODEL 2)
# =========================================
with tab2:
    st.markdown("### 📳 High-Frequency Vibration Analysis (NASA Bearing Dataset)")
    
    col_v1, col_v2 = st.columns()
    
    with col_v1:
        st.markdown("<div style='background-color:#fff3e0; padding:15px; border-radius:10px; border: 1px solid #ffb74d;'>", unsafe_allow_html=True)
        st.markdown("#### 🎚️ Physical Degradation Simulator")
        sim_level = st.slider("Bearing Degradation Phase (%)", 0, 100, 10, help="0%: Brand New | 50%: Early Wear | 85%: Warning | 100%: Total Failure")
        
        # Physics Simulator logic
        rms_val = 0.05 + (sim_level/100)**2 * 0.20
        kurt_val = 3.0 + (sim_level/100)**1.5 * 5.0
        skew_val = (sim_level/100) * 0.5
        p2p_val = rms_val * 4.5
        crest_val = p2p_val / rms_val if rms_val>0 else 0
        
        st.markdown("**(Simulated Sensor Outputs):**")
        st.caption(f"RMS: {rms_val:.3f} | Kurtosis: {kurt_val:.2f} | Skewness: {skew_val:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_v2:
        raw_df2 = pd.DataFrame({'RMS':[rms_val], 'Kurtosis':[kurt_val], 'Skewness':[skew_val], 'Peak_to_Peak':[p2p_val], 'Crest_Factor':[crest_val]})
        processed_df2 = scaler2.transform(raw_df2)
        probs2 = np.array(model2.predict_proba(processed_df2)).flatten()
        
        # Map NASA classes [0: Normal, 1: Early, 2: Warn, 3: Fail]
        p_b_norm = probs2*100 if len(probs2)>0 else 100
        p_b_early = probs2*100 if len(probs2)>1 else 0
        p_b_warn = probs2*100 if len(probs2)>2 else 0
        p_b_fail = probs2*100 if len(probs2)>3 else 0
        
        # Calculate Bearing Risk (Weighted Sum for smooth transition)
        bearing_overall_risk = (p_b_early*0.4) + (p_b_warn*0.8) + (p_b_fail*1.0)
        
        # Save to session
        st.session_state.bearing_risk = bearing_overall_risk
        if bearing_overall_risk < 20: st.session_state.bearing_status = "Healthy"
        elif bearing_overall_risk < 60: st.session_state.bearing_status = "Micro-cracks Detected"
        else: st.session_state.bearing_status = "Severe Spalling/Failure"

        c3, c4 = st.columns(2)
        with c3: st.plotly_chart(create_industrial_gauge(bearing_overall_risk, "Bearing Risk Score", "#9B59B6", height=250), use_container_width=True)
        with c4:
            st.markdown("#### AI Classification Confidence")
            df_b_faults = pd.DataFrame({'Phase': ['Normal', 'Early Wear', 'Warning', 'Failure'], 'Prob': [p_b_norm, p_b_early, p_b_warn, p_b_fail]})
            fig_b_bar = px.bar(df_b_faults, x='Prob', y='Phase', orientation='h', color='Prob', color_continuous_scale='Purples', range_x=)
            fig_b_bar.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_b_bar, use_container_width=True)

# =========================================
# TAB 3: SYSTEM FUSION (CUMULATIVE RELIABILITY)
# =========================================
with tab3:
    st.markdown("### 🔗 Master Asset Fusion (System of Systems)")
    
    # Mathematical Fusion Logic
    m_risk_frac = st.session_state.motor_risk / 100.0
    b_risk_frac = st.session_state.bearing_risk / 100.0
    
    # Cumulative Reliability Equation
    integrated_risk_frac = 1.0 - ((1.0 - m_risk_frac) * (1.0 - b_risk_frac))
    integrated_risk_pct = integrated_risk_frac * 100.0
    
    sys_color = "#00A65A" if integrated_risk_pct < 40 else ("#F39C12" if integrated_risk_pct < 75 else "#DD4B39")
    
    fc1, fc2, fc3 = st.columns([1, 0.2, 1])
    with fc1:
        st.markdown(f"<div class='metric-card' style='border-top-color:#3498DB;'><h4>Motor Risk</h4><h1 style='color:#3498DB;'>{st.session_state.motor_risk:.1f}%</h1></div>", unsafe_allow_html=True)
    with fc2:
        st.markdown("<h1 style='text-align:center; margin-top:30px; color:#aaa;'>✖</h1>", unsafe_allow_html=True)
    with fc3:
        st.markdown(f"<div class='metric-card' style='border-top-color:#9B59B6;'><h4>Bearing Risk</h4><h1 style='color:#9B59B6;'>{st.session_state.bearing_risk:.1f}%</h1></div>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.plotly_chart(create_industrial_gauge(integrated_risk_pct, "MASTER INTEGRATED ASSET RISK", sys_color, height=350), use_container_width=True)

# =========================================
# TAB 4: EXECUTIVE COMMAND CENTER
# =========================================
with tab4:
    st.markdown("### 📊 C-Level Executive Overview")
    
    health_score = 100.0 - integrated_risk_pct
    health_color = "#00A65A" if health_score > 60 else ("#F39C12" if health_score > 25 else "#DD4B39")
    
    # Financial Impact Logic (Mock logic for presentation)
    base_asset_value = 150000 # $150k machine
    downtime_cost_per_hr = 5000
    risk_factor = integrated_risk_pct / 100.0
    value_at_risk = base_asset_value * risk_factor
    
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: {health_color}; background-color: #2D2D2D; color: white;">
            <h4 style="color:#aaa;">Plant Health Score</h4>
            <h1 style="color:{health_color}; font-size:48px; margin:0;">{health_score:.1f} / 100</h1>
            <p style="color:#888;">System-wide Asset Vitality</p>
        </div>
        """, unsafe_allow_html=True)
    with ex2:
         st.markdown(f"""
        <div class="metric-card" style="border-top-color: #E74C3C;">
            <h4 style="color:#666;">Financial Value at Risk</h4>
            <h1 style="color:#E74C3C; font-size:40px; margin:0;">${value_at_risk:,.0f}</h1>
            <p style="color:#888;">Potential Capex Loss</p>
        </div>
        """, unsafe_allow_html=True)
    with ex3:
        action_text = "Optimal Production" if integrated_risk_pct < 40 else ("Schedule Maintenance" if integrated_risk_pct < 75 else "EMERGENCY SHUTDOWN")
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #F1C40F;">
            <h4 style="color:#666;">AI Recommended Action</h4>
            <h2 style="color:#333; margin-top:10px;">{action_text}</h2>
            <p style="color:#888;">Primary Cause: {'Bearing Wear' if st.session_state.bearing_risk > st.session_state.motor_risk else 'Motor Stress'}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Dynamic Radar Chart for full system view
    st.markdown("#### 🎯 Holistic Risk Distribution Map")
    categories = ['Heat (HDF)', 'Strain (OSF)', 'Power (PWF)', 'Tool Wear (TWF)', 'Bearing Wear (Vib)']
    values = [
        st.session_state.motor_faults['HDF'],
        st.session_state.motor_faults['OSF'],
        st.session_state.motor_faults['PWF'],
        st.session_state.motor_faults['TWF'],
        st.session_state.bearing_risk
    ]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(221, 75, 57, 0.2)',
        line=dict(color='#DD4B39', width=2),
        name='Risk Profile'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=)),
        showlegend=False,
        height=400,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# =========================================
# 6. FOOTER
# =========================================
st.markdown("""
<br><hr>
<div style='text-align: center; color: #888888; font-size: 11px; font-family: sans-serif; line-height: 1.5;'>
    <strong>© 2026 MotorGuard APM | Developed by Eng. Mahmoud Reda Ibrahim for DBA Research.</strong><br>
    Integrating Machine Learning with System Reliability Theory for Enterprise Asset Performance Management.
</div>
""", unsafe_allow_html=True)