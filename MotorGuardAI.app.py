# ==============================================================================
# © 2026 MotorGuard AI 
# Predictive Maintenance & Fault Diagnostics Dashboard
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import datetime

# =========================================
# 1. إعدادات الصفحة الأساسية
# =========================================
st.set_page_config(page_title="MotorGuard AI", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

# تهيئة سجل الفحص في الذاكرة المؤقتة (Session State)
if 'inspection_log' not in st.session_state:
    st.session_state['inspection_log'] = pd.DataFrame(columns=['Timestamp', 'Motor Type', 'RPM', 'Torque (Nm)', 'Temp [K]', 'Diagnosis', 'Status'])

# =========================================
# 2. تحميل نماذج الذكاء الاصطناعي
# =========================================
@st.cache_resource
def load_ai_models():
    try:
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('motorguard_model.pkl')
        return scaler, model
    except Exception as e:
        st.error(f"⚠️ خطأ في تحميل ملفات الذكاء الاصطناعي. تأكد من وجود scaler.pkl و motorguard_model.pkl في نفس المجلد. التفاصيل: {e}")
        return None, None

scaler, model = load_ai_models()

# =========================================
# 3. تصميم الواجهة الرئيسية (Header)
# =========================================
st.markdown("""
<div style="background-color:#1E1E1E; padding:20px; border-radius:10px; border-left: 5px solid #00A65A;">
    <h2 style="color:white; margin:0;">⚙️ MotorGuard AI - Predictive Maintenance System</h2>
    <p style="color:#B0B0B0; margin:0;">نظام التشخيص المبكر لأعطال المحركات الصناعية باستخدام الذكاء الاصطناعي</p>
</div>
<br>
""", unsafe_allow_html=True)

# =========================================
# 4. لوحة التحكم الجانبية (Sidebar Inputs)
# =========================================
with st.sidebar:
    st.header("🎛️ SCADA Inputs")
    st.markdown("أدخل قراءات الحساسات اللحظية:")
    
    motor_type = st.selectbox("Motor Quality (Type)", ['L (Low)', 'M (Medium)', 'H (High)'], index=1)
    
    st.markdown("---")
    st.subheader("🌡️ Thermal Sensors")
    air_temp = st.number_input("Air Temperature [K]", min_value=290.0, max_value=310.0, value=298.0, step=0.1)
    process_temp = st.number_input("Process Temperature [K]", min_value=300.0, max_value=320.0, value=308.0, step=0.1)
    
    st.markdown("---")
    st.subheader("⚙️ Mechanical Sensors")
    rpm = st.slider("Rotational Speed [rpm]", min_value=1100, max_value=2900, value=1500, step=10)
    torque = st.slider("Torque [Nm]", min_value=10.0, max_value=80.0, value=40.0, step=0.5)
    tool_wear = st.slider("Tool Wear [min]", min_value=0, max_value=260, value=50, step=1)
    
    analyze_btn = st.button("🔍 Run Diagnostics", use_container_width=True, type="primary")

# =========================================
# 5. دوال مساعدة لرسم العدادات (Gauges)
# =========================================
def create_gauge(value, title, max_val, danger_threshold):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [None, max_val]},
            'bar': {'color': "#00A65A" if value < danger_threshold else "#DD4B39"},
            'steps': [
                {'range': [0, danger_threshold], 'color': "rgba(0, 166, 90, 0.1)"},
                {'range': [danger_threshold, max_val], 'color': "rgba(221, 75, 57, 0.1)"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': danger_threshold}
        }))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# =========================================
# 6. منطقة العرض والتحليل (Main Dashboard)
# =========================================
if analyze_btn and scaler and model:
    # تجهيز البيانات
    type_dict = {'L (Low)': 0, 'M (Medium)': 1, 'H (High)': 2}
    
    input_df = pd.DataFrame({
        'Type': [type_dict[motor_type]],
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rpm],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear]
    })
    
    # توحيد المقاييس (Scaling)
    cols_to_scale = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    
    # التنبؤ
    with st.spinner('يتم تحليل الإشارات واستخراج الأنماط...'):
        prediction = model.predict(input_df)
    
    # -----------------------------------------
    # عرض نتيجة التشخيص (Diagnosis Banner)
    # -----------------------------------------
    if prediction == 'Normal':
        status_color = "#00A65A"
        status_bg = "#E8F5E9"
        status_text = "✅ NORMAL - حالة المحرك مستقرة"
        alert_msg = "لا توجد مؤشرات لأعطال قريبة. جميع القراءات ضمن النطاق الآمن."
        sys_status = "Safe"
    else:
        status_color = "#DD4B39"
        status_bg = "#FDEDEC"
        status_text = f"🚨 ALERT - توقع حدوث عطل!"
        alert_msg = f"الذكاء الاصطناعي يحذر من حدوث: **{prediction}**. يُرجى فحص المعدة فوراً."
        sys_status = "Critical"

    st.markdown(f"""
    <div style="background-color:{status_bg}; padding:20px; border-radius:10px; border: 2px solid {status_color};">
        <h3 style="color:{status_color}; margin:0;">{status_text}</h3>
        <p style="font-size: 16px; margin-top: 5px;">{alert_msg}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # -----------------------------------------
    # عرض العدادات اللحظية (Gauges)
    # -----------------------------------------
    st.markdown("### 📊 Live Sensor Telemetry")
    g1, g2, g3 = st.columns(3)
    
    with g1:
        st.plotly_chart(create_gauge(rpm, "Speed (RPM)", 3000, 2500), use_container_width=True)
    with g2:
        st.plotly_chart(create_gauge(torque, "Torque (Nm)", 80, 60), use_container_width=True)
    with g3:
        st.plotly_chart(create_gauge(tool_wear, "Tool Wear (min)", 260, 200), use_container_width=True)

    # -----------------------------------------
    # تحديث سجل الفحص (Log)
    # -----------------------------------------
    new_log = pd.DataFrame([{
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Motor Type': motor_type.split(' '),
        'RPM': rpm,
        'Torque (Nm)': torque,
        'Temp [K]': process_temp,
        'Diagnosis': prediction,
        'Status': sys_status
    }])
    st.session_state['inspection_log'] = pd.concat([new_log, st.session_state['inspection_log']], ignore_index=True)

    # -----------------------------------------
    # استخراج التقرير (Download Report)
    # -----------------------------------------
    report_text = f"""
=========================================
 MotorGuard AI - Inspection Report
=========================================
Date/Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Motor Type: {motor_type}

--- Sensor Readings ---
Air Temp: {air_temp} K
Process Temp: {process_temp} K
Rotational Speed: {rpm} RPM
Torque: {torque} Nm
Tool Wear: {tool_wear} min

--- AI Diagnostics ---
Status: {sys_status.upper()}
Diagnosis: {prediction}
=========================================
Generated by MotorGuard AI Dashboard
    """
    st.download_button(
        label="📥 Download Diagnostic Report (TXT)",
        data=report_text,
        file_name=f"MotorGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

# =========================================
# 7. عرض سجل الفحوصات السابقة
# =========================================
st.markdown("---")
st.markdown("### 📋 Inspection History Log")
if not st.session_state['inspection_log'].empty:
    st.dataframe(st.session_state['inspection_log'], use_container_width=True)
else:
    st.info("لا توجد فحوصات مسجلة في الجلسة الحالية.")

# =========================================
# 8. التذييل (Footer)
# =========================================
st.markdown("""
<hr>
<div style='text-align: center; color: #888888; font-size: 12px;'>
    MotorGuard AI Engine V1.0 | Built with Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)