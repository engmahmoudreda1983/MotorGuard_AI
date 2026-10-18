import hmac
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import motorguard_core as mc

st.set_page_config(page_title="MotorGuard AI", page_icon="⚙️", layout="wide")

STRETCH = "stretch"
NAVY, BLUE = "#0b2e59", "#4da6ff"
GREEN, AMBER, RED = "#1a9e5c", "#d99a00", "#d63a3a"
STATUS_COLOR = {'EXCELLENT': GREEN, 'WATCH': AMBER, 'ACTION': RED}
STATUS_LABEL = {'EXCELLENT': 'NORMAL (Low Failure Risk)', 'WATCH': 'WATCH (Elevated Failure Risk)',
                'ACTION': 'ACTION REQUIRED (High Failure Risk)'}
MODE_ORDER = list(mc.MODES)
DEMO_DATE = '2026-01-31'

st.markdown("""
<style>
.pg-header {background:#0b2e59;padding:16px 22px;border-radius:10px;margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;gap:16px;}
.pg-header img {height:64px;background:#fff;border-radius:8px;padding:4px 8px;}
.pg-header h1 {color:#fff;margin:0;font-size:1.9rem;}
.pg-header p {color:#9fd0ff;margin:4px 0 0 0;font-size:1rem;}
.pg-banner {border:2px solid;border-radius:8px;padding:10px 14px;text-align:center;margin:6px 0 12px 0;}
.pg-banner h2 {margin:0;font-size:1.5rem;}
.pg-banner p {margin:2px 0 0 0;font-size:.95rem;}
.pg-note {background:rgba(77,166,255,.10);border-left:4px solid #4da6ff;padding:8px 12px;border-radius:4px;font-size:.92rem;margin-bottom:10px;}
.pg-card {border:1px solid #d5dce6;border-radius:10px;padding:12px 14px;text-align:center;}
.pg-card h4 {margin:0;color:#0b2e59;font-size:1.05rem;}
.pg-card .big {font-size:1.7rem;font-weight:700;margin:4px 0;}
.pg-card .sub {font-size:.85rem;color:#5a6575;}
.pg-foot {text-align:center;color:#888;font-size:12px;padding:10px;}
</style>
""", unsafe_allow_html=True)

FOOTER = """
<div class='pg-foot'><strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
<em>Confidential and Proprietary. The predictive algorithms and expert system overrides within this application are protected intellectual property.</em></div>
"""


def banner(title, sub, color):
    st.markdown(f"<div class='pg-banner' style='border-color:{color}'><h2 style='color:{color}'>{title}</h2>"
                f"<p>{sub}</p></div>", unsafe_allow_html=True)


def secrets_ok():
    try:
        return "APP_USERNAME" in st.secrets and "APP_PASSWORD" in st.secrets
    except Exception:
        return False


# ==========================================
# --- 1. Login ---
# ==========================================
for key, default in [('logged_in', False), ('register', []), ('hist', None)]:
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state['logged_in']:
    st.markdown("<br><br><h1 style='text-align:center;color:#4da6ff;'>⚙️ MotorGuard AI Login</h1>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.warning("🔒 Restricted Access. Authorized Personnel Only.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Secure Login", width=STRETCH):
                if not secrets_ok():
                    st.error("Login is not configured: add APP_USERNAME and APP_PASSWORD to the app's Secrets.")
                elif hmac.compare_digest(username, st.secrets["APP_USERNAME"]) and \
                        hmac.compare_digest(password, st.secrets["APP_PASSWORD"]):
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password")
    st.markdown(FOOTER, unsafe_allow_html=True)
    st.stop()

# ==========================================
# --- 2. Header and model ---
# ==========================================
_, col_r = st.columns([10, 1])
with col_r:
    if st.button("Logout 🚪"):
        st.session_state['logged_in'] = False
        st.rerun()


@st.cache_data
def logo_b64():
    try:
        import base64, io
        from PIL import Image
        im = Image.open('logo.png')
        im.thumbnail((360, 200))
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


_logo = logo_b64()
_logo_html = f"<img src='data:image/png;base64,{_logo}' alt='PowerGuard AI'>" if _logo else ""
st.markdown(f"""
<div class='pg-header'>
  <div>
    <h1>⚙️ MotorGuard AI: HV Motor Condition and Risk Monitor</h1>
    <p>Stator, cooling, bearings, vibration, supply, rotor and starting duty, fused into one risk per motor</p>
  </div>
  {_logo_html}
</div>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the thermal model…")
def load_model():
    try:
        with open('thermal_model.pkl', 'rb') as f:
            m = pickle.load(f)
        m.predict(np.array([[80.0, 1.0, 35.0]]))
        return m
    except Exception:
        # model file missing or saved with another scikit-learn version: train it again (a few seconds)
        from sklearn.ensemble import RandomForestRegressor
        X, y = mc.make_thermal_training()
        return RandomForestRegressor(n_estimators=60, max_depth=11, min_samples_leaf=8, random_state=0,
                                     n_jobs=-1).fit(X[:16000], y[:16000])


MODEL = load_model()


@st.cache_data(show_spinner="Assessing every reading…")
def run_series(online, register, tests):
    return mc.assess_series(online, register, MODEL, tests)


@st.cache_data(show_spinner=False)
def demo_full():
    return mc.demo_data()


def demo_series():
    d = demo_full()
    return run_series(d['online'], d['register'], d['tests'])


# ==========================================
# --- 3. Shared charts and blocks ---
# ==========================================
def gauge(value, color, title='Failure Risk (%)', height=290):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), number={'suffix': '%', 'font': {'size': 34}, 'valueformat': '.0f'},
        title={'text': title, 'font': {'size': 15}}, domain={'x': [0, 1], 'y': [0, 0.85]},
        gauge={'axis': {'range': [0, 100], 'tickvals': [0, 35, 70, 100], 'tickfont': {'size': 12}},
               'bar': {'color': color, 'thickness': 0.3},
               'steps': [{'range': [0, 35], 'color': 'rgba(26,158,92,.25)'},
                         {'range': [35, 70], 'color': 'rgba(217,154,0,.25)'},
                         {'range': [70, 100], 'color': 'rgba(214,58,58,.25)'}]}))
    fig.update_layout(height=height, margin=dict(l=35, r=35, t=40, b=10))
    return fig


def band_color(v):
    return GREEN if v <= mc.EXCELLENT_MAX else AMBER if v <= mc.WATCH_MAX else RED


def mode_plot(risks, title=None, height=300):
    names = sorted(risks, key=lambda k: risks[k])
    vals = [risks[n] for n in names]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation='h', marker_color=[band_color(v) for v in vals],
                           text=[f"{v:.0f}%" for v in vals], textposition='outside', cliponaxis=False,
                           hovertemplate='%{y}: %{x:.1f}%<extra></extra>'))
    for x in (35, 70):
        fig.add_vline(x=x, line_dash='dash', line_color='#888')
    fig.update_layout(title=title or 'Risk by failure mode (risk drivers)', height=height,
                      xaxis=dict(range=[0, 115], title='Risk (%)'), margin=dict(l=10, r=20, t=45, b=40))
    return fig


def rpn_plot(a, height=230):
    rpn = int(a['RPN'])
    col = GREEN if rpn < 40 else AMBER if rpn < 100 else RED
    fig = go.Figure()
    for x0, x1, c, t in [(0, 40, 'rgba(26,158,92,.18)', 'Low'), (40, 100, 'rgba(217,154,0,.18)', 'Medium'),
                         (100, 200, 'rgba(214,58,58,.18)', 'High')]:
        fig.add_shape(type='rect', x0=x0, x1=x1, y0=-0.5, y1=0.5, fillcolor=c, line_width=0, layer='below')
        fig.add_annotation(x=(x0 + x1) / 2, y=0.62, text=t, showarrow=False, font=dict(size=11, color='#555'))
    fig.add_bar(x=[rpn], y=[0], orientation='h', width=0.45, marker_color=col,
                text=[f"RPN {rpn} = S {int(a['S'])} × O {int(a['O'])} × D {int(a['D'])}"],
                textposition='inside' if rpn > 110 else 'outside', insidetextanchor='middle', cliponaxis=False,
                hoverinfo='skip', textfont=dict(color='white' if rpn > 110 else '#222'))
    fig.update_layout(title='Risk Priority Number of the governing mode (max 200)', height=height, showlegend=False,
                      xaxis=dict(range=[0, 200], tickvals=[0, 40, 100, 200], title='RPN'),
                      yaxis=dict(visible=False, range=[-0.6, 0.8]), margin=dict(l=10, r=10, t=45, b=40))
    return fig


def fmea_rpn_plot(fmea, height=300):
    t = sorted(fmea, key=lambda m: m['RPN'])
    fig = go.Figure(go.Bar(x=[m['RPN'] for m in t], y=[m['Failure Mode'] for m in t], orientation='h',
                           marker_color=[GREEN if m['RPN'] < 40 else AMBER if m['RPN'] < 100 else RED for m in t],
                           text=[f"{m['RPN']}" for m in t], textposition='outside', cliponaxis=False,
                           hovertemplate='%{y}: RPN %{x}<extra></extra>'))
    for x in (40, 100):
        fig.add_vline(x=x, line_dash='dash', line_color='#888')
    fig.update_layout(title='RPN by failure mode', height=height, xaxis=dict(range=[0, 200], title='RPN'),
                      margin=dict(l=10, r=20, t=45, b=40))
    return fig


def thermal_plot(a, rise_class, height=320):
    """Expected stator temperature against load at today's ambient and unbalance, with the measured point."""
    loads = np.linspace(0, 120, 49)
    X = np.c_[loads, np.full_like(loads, a['Current Unbalance (%)']), np.full_like(loads, a['Ambient (C)'])]
    exp = a['Ambient (C)'] + MODEL.predict(X) * mc.RISE_RATED.get(rise_class, 105.0)
    fig = go.Figure()
    fig.add_scatter(x=loads, y=exp, mode='lines', name='Expected (thermal model)', line=dict(color=NAVY))
    fig.add_scatter(x=loads, y=exp + mc.COOLING_DEV_WATCH, mode='lines', name=f'Expected + {mc.COOLING_DEV_WATCH:.0f} °C',
                    line=dict(color=AMBER, dash='dot'))
    fig.add_hline(y=a['Stator Alarm (C)'], line_dash='dash', line_color=AMBER, annotation_text='Alarm', annotation_position='bottom left')
    fig.add_hline(y=a['Stator Trip (C)'], line_dash='dash', line_color=RED, annotation_text='Trip', annotation_position='top left')
    fig.add_scatter(x=[a['Effective Load (%)']], y=[a['Stator Temp (C)']], mode='markers', name='Measured',
                    marker=dict(size=14, color=band_color(max(a['Mode Risks']['Cooling'], a['Mode Risks']['Stator insulation'])),
                                line=dict(color='#222', width=1)))
    fig.update_layout(title='Stator temperature: measured vs expected', height=height, xaxis_title='Load after derating (%)',
                      yaxis_title='°C', legend=dict(orientation='h', y=-0.25), margin=dict(l=40, r=10, t=45, b=40))
    return fig


def iso_plot(a, height=230):
    ab, bc, cd = a['Zone Bounds']
    top = max(cd * 1.5, a['Vibration Max (mm/s)'] * 1.15)
    fig = go.Figure()
    for x0, x1, c, t in [(0, ab, 'rgba(26,158,92,.30)', 'A'), (ab, bc, 'rgba(26,158,92,.12)', 'B'),
                         (bc, cd, 'rgba(217,154,0,.22)', 'C'), (cd, top, 'rgba(214,58,58,.22)', 'D')]:
        fig.add_shape(type='rect', x0=x0, x1=x1, y0=-0.5, y1=1.5, fillcolor=c, line_width=0, layer='below')
        fig.add_annotation(x=(x0 + x1) / 2, y=1.75, text=f"Zone {t}", showarrow=False, font=dict(size=11, color='#555'))
    fig.add_bar(x=[a['Vibration DE (mm/s)'], a['Vibration NDE (mm/s)']], y=['DE', 'NDE'], orientation='h', width=0.5,
                marker_color=NAVY, text=[f"{a['Vibration DE (mm/s)']:.1f}", f"{a['Vibration NDE (mm/s)']:.1f}"],
                textposition='outside', cliponaxis=False)
    fig.update_layout(title=f"Vibration, ISO 20816-3 ({a['ISO Group'].lower()})", height=height, showlegend=False,
                      xaxis=dict(range=[0, top], title='mm/s r.m.s.'), yaxis=dict(range=[-0.6, 2.0]),
                      margin=dict(l=10, r=10, t=45, b=40))
    return fig


def bearing_plot(a, btype, height=230):
    ba, bt = mc.BEARING_LIMITS[btype]
    v = [a['Bearing DE Temp (C)'], a['Bearing NDE Temp (C)']]
    fig = go.Figure(go.Bar(x=['DE', 'NDE'], y=v, marker_color=[GREEN if x < ba else AMBER if x < bt else RED for x in v],
                           text=[f"{x:.0f} °C" for x in v], textposition='outside', cliponaxis=False, width=0.5))
    fig.add_hline(y=ba, line_dash='dash', line_color=AMBER, annotation_text=f'Alarm {ba:.0f}', annotation_position='bottom left')
    fig.add_hline(y=bt, line_dash='dash', line_color=RED, annotation_text=f'Trip {bt:.0f}', annotation_position='top left')
    fig.update_layout(title=f'Bearing temperature ({btype.lower()})', height=height, yaxis=dict(range=[0, bt + 20], title='°C'),
                      margin=dict(l=40, r=10, t=45, b=30))
    return fig


def derate_plot(a, height=230):
    x = np.linspace(0, 5, 51)
    fig = go.Figure()
    fig.add_scatter(x=x, y=[mc.nema_derate(v) for v in x], mode='lines', line=dict(color=NAVY), name='NEMA MG1')
    fig.add_scatter(x=[a['Voltage Unbalance (%)']], y=[a['NEMA Derating']], mode='markers', marker=dict(size=13, color=band_color(a['Mode Risks']['Supply'])),
                    name='This motor')
    fig.update_layout(title='Derating for voltage unbalance (NEMA MG1)', height=height, showlegend=False,
                      xaxis_title='Voltage unbalance (%)', yaxis=dict(range=[0.7, 1.02], title='Derating factor'),
                      margin=dict(l=40, r=10, t=45, b=40))
    return fig


def fmea_table(a):
    return pd.DataFrame([{'Failure Mode': m['Failure Mode'], 'Risk (%)': round(m['Risk (%)']), 'S': m['S'], 'O': m['O'], 'D': m['D'],
                          'RPN': m['RPN'], 'RPN Level': m['RPN Level']} for m in a['FMEA']])


def show_result(a, reg, k):
    color = STATUS_COLOR[a['Status']]
    banner(STATUS_LABEL[a['Status']], f"{a['Motor ID']}  ·  {a['Service'] or 'HV motor'}  ·  governing mode: {a['Governing Mode']}", color)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Load", f"{a['Load (%)']:.0f} %")
    m2.metric("Stator temperature", f"{a['Stator Temp (C)']:.0f} °C",
              None if a['Stator Temp Excess (C)'] is None else f"{a['Stator Temp Excess (C)']:+.0f} °C vs expected", delta_color='inverse')
    m3.metric("Vibration (ISO zone)", f"{a['Vibration Max (mm/s)']:.1f} mm/s ({a['ISO Zone']})")
    m4.metric("Voltage unbalance", f"{a['Voltage Unbalance (%)']:.1f} %")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.plotly_chart(gauge(a['Overall Risk (%)'], color), width=STRETCH, key=f"{k}_g")
    with c2:
        st.plotly_chart(mode_plot(a['Mode Risks']), width=STRETCH, key=f"{k}_m")
    st.markdown("#### Why this result")
    for t in mc.findings(a):
        st.markdown(f"- {t}")
    st.warning(f"**Action:** {a['Action']}")
    with st.expander("📐 Engineering detail", expanded=False):
        e1, e2 = st.columns(2)
        e1.plotly_chart(thermal_plot(a, str(reg.get('Rise Class', 'B')).upper()[:1]), width=STRETCH, key=f"{k}_th")
        e2.plotly_chart(iso_plot(a), width=STRETCH, key=f"{k}_iso")
        e3, e4 = st.columns(2)
        btype = 'Sleeve' if str(reg.get('Bearing Type', 'Rolling')).lower().startswith('sl') else 'Rolling'
        e3.plotly_chart(bearing_plot(a, btype), width=STRETCH, key=f"{k}_b")
        e4.plotly_chart(derate_plot(a), width=STRETCH, key=f"{k}_d")
    st.markdown("#### 🧮 FMEA")
    f1, f2 = st.columns([1.2, 1])
    f1.dataframe(fmea_table(a), hide_index=True, width=STRETCH)
    f1.caption("RPN = Severity × Occurrence × Detection. S = severity of the failure mode; O = mode risk ÷ 10; "
               "D = 2 for online monitoring. Low < 40, Medium 40-99, High ≥ 100.")
    if a['Days to Zone D'] is not None:
        f1.metric("Estimated days to vibration zone D", "Already in zone D" if a['Days to Zone D'] == 0 else f"{a['Days to Zone D']} days")
    with f2:
        st.plotly_chart(rpn_plot(a), width=STRETCH, key=f"{k}_rpn")
        st.plotly_chart(fmea_rpn_plot(a['FMEA'], 280), width=STRETCH, key=f"{k}_fr")


REC_KEYS = ['Motor ID', 'Service', 'Rated kW', 'Load (%)', 'Stator Temp (C)', 'Stator Temp Excess (C)', 'Vibration Max (mm/s)',
            'ISO Zone', 'Voltage Unbalance (%)', 'Current Unbalance (%)', 'Governing Mode', 'Overall Risk (%)', 'Status',
            'S', 'O', 'D', 'RPN', 'RPN Level', 'Days to Zone D', 'Action', 'Mode Risks', 'FMEA']


def to_record(a, source):
    rec = {k: a.get(k) for k in REC_KEYS}
    rec['Source'] = source
    rec['Assessed'] = datetime.now().strftime('%Y-%m-%d %H:%M') if source == 'Motor page' else pd.Timestamp(a['Timestamp']).strftime('%Y-%m-%d')
    rec['Findings'] = mc.findings(a)
    return rec


@st.cache_data(show_spinner="Analyzing the demo plant…")
def demo_records():
    rows = [a for a in demo_series() if a['Timestamp'] <= pd.Timestamp(DEMO_DATE) + pd.Timedelta(days=1)]
    return [to_record(a, 'Demo plant') for a in mc.latest(rows)]


@st.cache_data(show_spinner=False)
def demo_snapshot():
    """Inputs of every demo motor on the demo date (used to fill the motor page)."""
    rows = [a for a in demo_series() if a['Timestamp'] <= pd.Timestamp(DEMO_DATE) + pd.Timedelta(days=1)]
    d = demo_full()
    on = d['online']
    out = {}
    for a in mc.latest(rows):
        r = on[(on['Motor ID'] == a['Motor ID']) & (pd.to_datetime(on['Timestamp']) <= pd.Timestamp(a['Timestamp']))].iloc[-1]
        out[a['Motor ID']] = dict(r=r.to_dict(), a=a)
    return out


SRC_PAGES, SRC_DEMO = "Results from the analysis pages", "Demo plant (synthetic)"


def current_records():
    if st.session_state.get('ov_src', SRC_PAGES) == SRC_DEMO:
        return demo_records()
    return list(st.session_state['register'])


def ranking_plot(R, height=None, top=20):
    t = R.sort_values('Overall Risk (%)', ascending=False).head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=t['Overall Risk (%)'], y=t['Motor ID'], orientation='h',
                           marker_color=[STATUS_COLOR[s] for s in t['Status']],
                           text=[f"{r:.0f}%  ·  {g}" for r, g in zip(t['Overall Risk (%)'], t['Governing Mode'])],
                           textposition='outside', cliponaxis=False, hovertemplate='%{y}: %{x:.1f}%<extra></extra>'))
    for x, tx in [(35, 'Watch'), (70, 'Action')]:
        fig.add_vline(x=x, line_dash='dash', line_color='#888', annotation_text=tx, annotation_position='top')
    fig.update_layout(title=f"Motor risk ranking (top {min(top, len(R))})", height=height or (140 + 34 * len(t)),
                      xaxis=dict(range=[0, 125], title='Failure Risk (%)'), yaxis=dict(type='category'),
                      margin=dict(l=10, r=30, t=60, b=40))
    return fig


def heat_plot(R, height=None):
    t = R.sort_values('Overall Risk (%)', ascending=False)
    z = [[r['Mode Risks'][m] for m in MODE_ORDER] for _, r in t.iterrows()]
    fig = go.Figure(go.Heatmap(z=z, x=MODE_ORDER, y=list(t['Motor ID']), zmin=0, zmax=100,
                               colorscale=[[0, '#e8f5ee'], [0.35, '#bfe3cd'], [0.351, '#fbe8b3'], [0.7, '#f2c14e'],
                                           [0.701, '#f2a6a6'], [1, RED]],
                               text=[[f"{v:.0f}" for v in row] for row in z], texttemplate='%{text}',
                               hovertemplate='%{y} · %{x}: %{z:.0f}%<extra></extra>', colorbar=dict(title='%', thickness=12)))
    fig.update_layout(title='Risk by motor and failure mode', height=height or (150 + 34 * len(t)),
                      yaxis=dict(type='category', autorange='reversed'), xaxis=dict(tickangle=-25),
                      margin=dict(l=10, r=10, t=50, b=60))
    return fig


def gov_plot(R, height=300):
    above = R[R['Status'] != 'EXCELLENT']
    fig = go.Figure()
    for stt, name in [('WATCH', 'Watch'), ('ACTION', 'Action')]:
        fig.add_bar(x=MODE_ORDER, y=[int(((above['Governing Mode'] == m) & (above['Status'] == stt)).sum()) for m in MODE_ORDER],
                    name=name, marker_color=STATUS_COLOR[stt])
    fig.update_layout(barmode='stack', title='Motors above normal by governing failure mode', height=height,
                      yaxis=dict(title='Motors', dtick=1), xaxis=dict(tickangle=-25), legend=dict(orientation='h', y=-0.45),
                      margin=dict(l=40, r=10, t=50, b=40))
    return fig


# ==========================================
# --- 4. Pages ---
# ==========================================
DEF = {'id': 'M-101', 'service': 'Pump drive', 'kw': 1250.0, 'irated': 140.0, 'kv': 6.6, 'ins': 'F', 'rise': 'B', 'btype': 'Rolling',
       'found': 'Rigid', 'allowed': 3, 'tstart': 8.0,
       'ia': 105.0, 'ib': 104.0, 'ic': 106.0, 'vab': 6.60, 'vbc': 6.58, 'vca': 6.62, 'amb': 35.0, 'ts': 92.0,
       'bde': 72.0, 'bnde': 69.0, 'vde': 1.8, 'vnde': 1.5, 'starts': 1, 'tst': 8.2,
       'vtr': 0.0, 'btr': 0.0, 'ir': 0.0, 'pi': 0.0, 'mcsa': 0.0}
INS = ['B', 'F', 'H']


def fill_example(prefix):
    mid = st.session_state.get(f'{prefix}_ex')
    if not mid or mid == '—':
        return
    s = demo_snapshot()[mid]; r, a = s['r'], s['a']
    reg = demo_full()['register'].set_index('Motor ID').loc[mid]
    ss = st.session_state
    vals = {'id': mid, 'service': reg['Service'], 'kw': float(reg['Rated kW']), 'irated': float(reg['Rated Current (A)']),
            'kv': float(reg['Rated Voltage (kV)']), 'ins': reg['Insulation Class'], 'rise': reg['Rise Class'],
            'btype': reg['Bearing Type'], 'found': reg['Foundation'], 'allowed': int(reg['Allowed Starts per Day']),
            'tstart': float(reg['Rated Start Time (s)']),
            'ia': float(r['Ia']), 'ib': float(r['Ib']), 'ic': float(r['Ic']), 'vab': float(r['Vab']), 'vbc': float(r['Vbc']),
            'vca': float(r['Vca']), 'amb': round(float(a['Ambient (C)']), 1), 'ts': round(float(a['Stator Temp (C)']), 1),
            'bde': round(float(a['Bearing DE Temp (C)']), 1), 'bnde': round(float(a['Bearing NDE Temp (C)']), 1),
            'vde': round(float(a['Vibration DE (mm/s)']), 2), 'vnde': round(float(a['Vibration NDE (mm/s)']), 2),
            'starts': int(a['Starts']), 'tst': float(a['Max Start Time (s)']),
            'vtr': 0.0, 'btr': 0.0, 'ir': 0.0, 'pi': 0.0, 'mcsa': 0.0}
    # tests and trends as the series saw them on that date
    t = a.get('Tests', {})
    vals.update({'ir': float(t.get('IR 1min (MOhm)') or 0), 'pi': float(t.get('PI') or 0), 'mcsa': float(t.get('MCSA Sideband (dB)') or 0)})
    tr = demo_trends().get(mid, {})
    vals.update({'vtr': round(tr.get('v', 0.0), 2), 'btr': round(tr.get('b', 0.0), 1)})
    for k, v in vals.items():
        ss[f'{prefix}_{k}'] = v


@st.cache_data(show_spinner=False)
def demo_trends():
    """60-day trends of each demo motor on the demo date (same calculation as the series)."""
    d = demo_full(); on = d['online'].copy(); on['Timestamp'] = pd.to_datetime(on['Timestamp'])
    on = on[on['Timestamp'] <= pd.Timestamp(DEMO_DATE) + pd.Timedelta(days=1)]
    out = {}
    for mid, g in on.groupby('Motor ID'):
        vmax = np.maximum(g['Vibration DE (mm/s)'], g['Vibration NDE (mm/s)']).to_numpy()
        bmax = np.maximum(g['Bearing DE Temp (C)'], g['Bearing NDE Temp (C)']).to_numpy()
        out[mid] = {'v': float(mc._rolling(g['Timestamp'], vmax, mc.TREND_WINDOW_D, 'slope')[-1]),
                    'b': float(mc._rolling(g['Timestamp'], bmax, mc.TREND_WINDOW_D, 'slope')[-1])}
    return out


def init_defaults(prefix):
    for k, v in DEF.items():
        st.session_state.setdefault(f'{prefix}_{k}', v)


def nameplate(p, short=False):
    st.markdown("**Motor data (nameplate)**")
    c1, c2 = st.columns(2)
    c1.text_input("Motor ID", key=f'{p}_id')
    c2.text_input("Service", key=f'{p}_service')
    c1, c2, c3 = st.columns(3)
    c1.number_input("Rated power (kW)", min_value=1.0, key=f'{p}_kw')
    c2.number_input("Rated current (A)", min_value=1.0, key=f'{p}_irated')
    c3.number_input("Rated voltage (kV)", min_value=0.4, key=f'{p}_kv')
    if short == 'thermal':
        c1, c2 = st.columns(2)
        c1.selectbox("Insulation class", INS, key=f'{p}_ins')
        c2.selectbox("Temperature-rise class", INS, key=f'{p}_rise', help="Rated rise the motor was designed for (often B in a class F motor).")
    elif short == 'mech':
        c1, c2 = st.columns(2)
        c1.selectbox("Bearing type", ['Rolling', 'Sleeve'], key=f'{p}_btype')
        c2.selectbox("Foundation", ['Rigid', 'Flexible'], key=f'{p}_found', help="ISO 20816-3 support class.")
    elif short == 'supply':
        c1, c2 = st.columns(2)
        c1.number_input("Allowed starts per day", min_value=1, key=f'{p}_allowed')
        c2.number_input("Rated start time (s)", min_value=0.5, key=f'{p}_tstart', help="Acceleration time at commissioning.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Insulation class", INS, key=f'{p}_ins')
        c2.selectbox("Rise class", INS, key=f'{p}_rise')
        c3.selectbox("Bearing type", ['Rolling', 'Sleeve'], key=f'{p}_btype')
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Foundation", ['Rigid', 'Flexible'], key=f'{p}_found')
        c2.number_input("Allowed starts / day", min_value=1, key=f'{p}_allowed')
        c3.number_input("Rated start time (s)", min_value=0.5, key=f'{p}_tstart')


def currents(p):
    st.markdown("**Phase currents (A)**")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Ia", min_value=0.0, key=f'{p}_ia'); c2.number_input("Ib", min_value=0.0, key=f'{p}_ib')
    c3.number_input("Ic", min_value=0.0, key=f'{p}_ic')


def voltages(p):
    st.markdown("**Line voltages (kV)**")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Vab", min_value=0.0, format="%.3f", key=f'{p}_vab'); c2.number_input("Vbc", min_value=0.0, format="%.3f", key=f'{p}_vbc')
    c3.number_input("Vca", min_value=0.0, format="%.3f", key=f'{p}_vca')


def thermal_inputs(p):
    st.markdown("**Temperatures (°C)**")
    c1, c2 = st.columns(2)
    c1.number_input("Ambient / cooling air in", key=f'{p}_amb')
    c2.number_input("Stator winding (hottest RTD)", key=f'{p}_ts')


def mech_inputs(p):
    st.markdown("**Bearings and vibration**")
    c1, c2 = st.columns(2)
    c1.number_input("Bearing DE temp (°C)", key=f'{p}_bde'); c2.number_input("Bearing NDE temp (°C)", key=f'{p}_bnde')
    c1, c2 = st.columns(2)
    c1.number_input("Vibration DE (mm/s r.m.s.)", min_value=0.0, key=f'{p}_vde'); c2.number_input("Vibration NDE (mm/s r.m.s.)", min_value=0.0, key=f'{p}_vnde')
    c1, c2 = st.columns(2)
    c1.number_input("Vibration trend (mm/s per month)", key=f'{p}_vtr', help="Change over the last 2 months. 0 if not known.")
    c2.number_input("Bearing temp trend (°C per month)", key=f'{p}_btr')


def start_inputs(p):
    st.markdown("**Starting**")
    c1, c2 = st.columns(2)
    c1.number_input("Starts today", min_value=0, key=f'{p}_starts'); c2.number_input("Longest start time today (s, 0 = none)", min_value=0.0, key=f'{p}_tst')


def test_inputs(p, which=('ir', 'pi', 'mcsa')):
    st.markdown("**Periodic tests (0 = not tested)**")
    cols = st.columns(len(which))
    lab = {'ir': ("IR 1 min (MΩ)", "Insulation resistance at 1 minute, corrected to 40 °C (IEEE 43)."),
           'pi': ("Polarization index", "IR 10 min ÷ IR 1 min (IEEE 43)."),
           'mcsa': ("MCSA sideband (dB)", "Rotor-bar sideband below the supply line in the current spectrum. Higher is better.")}
    for c, w in zip(cols, which):
        c.number_input(lab[w][0], min_value=0.0, key=f'{p}_{w}', help=lab[w][1])


def collect(p):
    ss = st.session_state
    g = lambda k: ss.get(f'{p}_{k}', DEF[k])
    reg = {'Motor ID': (g('id') or 'M-101').strip(), 'Service': g('service'), 'Rated kW': g('kw'), 'Rated Current (A)': g('irated'),
           'Rated Voltage (kV)': g('kv'), 'Insulation Class': g('ins'), 'Rise Class': g('rise'), 'Bearing Type': g('btype'),
           'Foundation': g('found'), 'Allowed Starts per Day': g('allowed'), 'Rated Start Time (s)': g('tstart')}
    rd = {'Ia': g('ia'), 'Ib': g('ib'), 'Ic': g('ic'), 'Vab': g('vab'), 'Vbc': g('vbc'), 'Vca': g('vca'), 'Ambient (C)': g('amb'),
          'Stator Temp (C)': g('ts'), 'Bearing DE Temp (C)': g('bde'), 'Bearing NDE Temp (C)': g('bnde'),
          'Vibration DE (mm/s)': g('vde'), 'Vibration NDE (mm/s)': g('vnde'), 'Starts': g('starts'), 'Max Start Time (s)': g('tst')}
    tests = {'IR 1min (MOhm)': g('ir') or None, 'PI': g('pi') or None, 'MCSA Sideband (dB)': g('mcsa') or None}
    trend = {'Vibration Trend (mm/s/30d)': g('vtr'), 'Bearing Temp Trend (C/30d)': g('btr')}
    return rd, reg, tests, trend


def example_picker(p):
    opts = ['—'] + list(demo_full()['register']['Motor ID'])
    svc = demo_full()['register'].set_index('Motor ID')['Service'].to_dict()
    st.selectbox("Fill with a demo motor (optional)", opts, key=f'{p}_ex', on_change=fill_example, args=(p,),
                 format_func=lambda m: m if m == '—' else f"{m} · {svc[m]} ({DEMO_DATE})")


def motor_page():
    p = 'mp'
    init_defaults(p)
    left, right = st.columns([1, 1.8])
    with left:
        st.markdown("#### ⚙️ Motor: latest online reading")
        example_picker(p)
        nameplate(p)
        currents(p); voltages(p); thermal_inputs(p); mech_inputs(p); start_inputs(p); test_inputs(p)
        go_btn = st.button("🔍 Analyze Motor", width=STRETCH, key=f"{p}_go", type='primary')
    with right:
        if go_btn:
            rd, reg, tests, trend = collect(p)
            a = mc.assess(rd, reg, MODEL, tests, trend)
            st.session_state[f"{p}_last"] = (a, reg)
            st.session_state['register'] = [r for r in st.session_state['register'] if r['Motor ID'] != a['Motor ID']]
            st.session_state['register'].append(to_record(a, 'Motor page'))
            st.rerun()  # the overview and the report are drawn earlier in the page: refresh them
        last = st.session_state.get(f"{p}_last")
        if last is None:
            st.markdown("<div class='pg-note'>Enter the latest reading of the motor on the left (or fill it with a demo motor) and "
                        "press <b>Analyze Motor</b>. The result is added to the Plant Overview and the Combined Report.</div>",
                        unsafe_allow_html=True)
        else:
            show_result(*last, p)
            st.caption("Added to the Plant Overview and the Combined Report.")


def check_page(p, kind):
    """Focused checks: only the inputs one subsystem needs. Full motor risk is on the Motor page."""
    init_defaults(p)
    left, right = st.columns([1, 1.8])
    title = {'thermal': '🌡️ Stator and cooling check', 'mech': '🔩 Bearings and vibration check', 'supply': '⚡ Supply and starting check'}[kind]
    with left:
        st.markdown(f"#### {title}")
        example_picker(p)
        nameplate(p, kind)
        if kind == 'thermal':
            currents(p); thermal_inputs(p); test_inputs(p, ('ir', 'pi'))
        elif kind == 'mech':
            st.number_input("Ambient (°C)", key=f'{p}_amb'); currents(p); mech_inputs(p)
        else:
            currents(p); voltages(p); start_inputs(p); test_inputs(p, ('mcsa',))
        go_btn = st.button("🔍 Analyze", width=STRETCH, key=f"{p}_go", type='primary')
    with right:
        if go_btn:
            rd, reg, tests, trend = collect(p)
            # inputs not on this page are set to neutral values so that only this subsystem is judged
            if kind != 'supply':
                rd.update({'Vab': 6.6, 'Vbc': 6.6, 'Vca': 6.6, 'Starts': 0, 'Max Start Time (s)': 0}); tests['MCSA Sideband (dB)'] = None
            if kind != 'mech':
                amb = rd['Ambient (C)']
                rd.update({'Bearing DE Temp (C)': amb, 'Bearing NDE Temp (C)': amb, 'Vibration DE (mm/s)': 0.5, 'Vibration NDE (mm/s)': 0.5})
                trend = {}
            if kind != 'thermal':
                rd['Stator Temp (C)'] = rd['Ambient (C)'] + 40; tests['PI'] = tests['IR 1min (MOhm)'] = None
            a = mc.assess(rd, reg, MODEL, tests, trend)
            if kind != 'thermal':
                a['Mode Risks']['Cooling'] = a['Mode Risks']['Stator insulation'] = 0.0
            st.session_state[f"{p}_last"] = (a, reg)
        last = st.session_state.get(f"{p}_last")
        if last is None:
            st.markdown("<div class='pg-note'>A quick check of one subsystem with only the inputs it needs. "
                        "For the full motor risk and FMEA use the <b>Motor Risk & FMEA</b> page.</div>", unsafe_allow_html=True)
            return
        a, reg = last
        modes = {'thermal': ['Stator insulation', 'Cooling'], 'mech': ['Bearing', 'Mechanical vibration'],
                 'supply': ['Supply', 'Rotor', 'Starting duty']}[kind]
        sub = {m: a['Mode Risks'][m] for m in modes}
        worst = max(sub, key=sub.get); st_ = mc.status_of(sub[worst])
        banner(STATUS_LABEL[st_], f"{a['Motor ID']}  ·  {worst} {sub[worst]:.0f}%", STATUS_COLOR[st_])
        c1, c2 = st.columns([1, 1.3])
        c1.plotly_chart(gauge(sub[worst], STATUS_COLOR[st_], f'{worst} risk (%)'), width=STRETCH, key=f"{p}_g")
        c2.plotly_chart(mode_plot(sub, 'Risk by failure mode', 240), width=STRETCH, key=f"{p}_m")
        if kind == 'thermal':
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected stator temp", "—" if a['Expected Stator Temp (C)'] is None else f"{a['Expected Stator Temp (C)']:.0f} °C")
            m2.metric("Measured − expected", "—" if a['Stator Temp Excess (C)'] is None else f"{a['Stator Temp Excess (C)']:+.0f} °C")
            m3.metric("Insulation ageing rate", f"{a['Ageing Factor']:.2f} × normal")
            st.plotly_chart(thermal_plot(a, str(reg['Rise Class']).upper()[:1]), width=STRETCH, key=f"{p}_th")
        elif kind == 'mech':
            m1, m2, m3 = st.columns(3)
            m1.metric("ISO 20816-3 zone", a['ISO Zone'])
            m2.metric("Bearing DE − NDE", f"{a['Bearing DE Temp (C)'] - a['Bearing NDE Temp (C)']:+.0f} °C")
            m3.metric("Days to zone D", "—" if a['Days to Zone D'] is None else ("now" if a['Days to Zone D'] == 0 else f"{a['Days to Zone D']}"))
            e1, e2 = st.columns(2)
            e1.plotly_chart(iso_plot(a), width=STRETCH, key=f"{p}_iso")
            btype = 'Sleeve' if str(reg['Bearing Type']).lower().startswith('sl') else 'Rolling'
            e2.plotly_chart(bearing_plot(a, btype), width=STRETCH, key=f"{p}_b")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Voltage unbalance", f"{a['Voltage Unbalance (%)']:.1f} %")
            m2.metric("Current unbalance", f"{a['Current Unbalance (%)']:.1f} %")
            m3.metric("Load after derating", f"{a['Effective Load (%)']:.0f} %")
            st.plotly_chart(derate_plot(a, 260), width=STRETCH, key=f"{p}_d")
        st.markdown("#### Why this result")
        for m in sorted(modes, key=lambda m: -sub[m]):
            st.markdown(f"- **{m} ({sub[m]:.0f}%):** " + ' '.join(a['Reasons'][m]))
        if st_ != 'EXCELLENT':
            st.warning(f"**Action:** {mc.ACTIONS[worst]}")
        st.caption("Quick check only: not added to the overview. Use the Motor Risk & FMEA page for the full assessment.")


# ==========================================
# --- 5. Tabs ---
# ==========================================
tabs = st.tabs(["🏭 Plant Overview", "⚙️ Motor Risk & FMEA", "🌡️ Stator & Cooling", "🔩 Bearings & Vibration",
                "⚡ Supply & Starting", "📄 Combined Report", "📈 Historical Study"])

NEED = {'register': ['Motor ID', 'Rated kW', 'Rated Current (A)', 'Insulation Class', 'Rise Class', 'Bearing Type', 'Foundation',
                     'Allowed Starts per Day', 'Rated Start Time (s)'],
        'online': ['Timestamp', 'Motor ID', 'Ia', 'Ib', 'Ic', 'Vab', 'Vbc', 'Vca', 'Ambient (C)', 'Stator Temp (C)',
                   'Bearing DE Temp (C)', 'Bearing NDE Temp (C)', 'Vibration DE (mm/s)', 'Vibration NDE (mm/s)', 'Starts', 'Max Start Time (s)'],
        'tests': ['Date', 'Motor ID'], 'events': ['Motor ID', 'Date', 'Event']}
FILE_LABEL = {'register': 'Motor register (required)', 'online': 'Online readings (required)',
              'tests': 'Periodic tests: IR, PI, MCSA (optional)', 'events': 'Events: failures and findings (for the study)'}


def read_uploads(prefix):
    files, errors = {}, []
    cols = st.columns(2)
    for i, k in enumerate(NEED):
        f = cols[i % 2].file_uploader(FILE_LABEL[k], type='csv', key=f"{prefix}_{k}")
        if f is None:
            files[k] = None
            continue
        try:
            df_ = pd.read_csv(f)
        except Exception as e:
            errors.append(f"{k}: cannot read the file ({e})")
            continue
        miss = [c for c in NEED[k] if c not in df_.columns]
        if miss:
            errors.append(f"{k}: missing columns {', '.join(miss[:8])}{' …' if len(miss) > 8 else ''}")
        files[k] = df_
    return files, errors


# ---------------- Overview
with tabs[0]:
    st.radio("Show", [SRC_PAGES, SRC_DEMO], horizontal=True, key='ov_src')
    R0 = current_records()
    if st.session_state['ov_src'] == SRC_PAGES:
        st.markdown("<div class='pg-note'>Every motor analyzed on the <b>Motor Risk & FMEA</b> page is collected here. "
                    "The latest result per motor is kept.</div>", unsafe_allow_html=True)
        if R0 and st.button("🗑 Clear results"):
            st.session_state['register'] = []
            st.rerun()
    else:
        st.info(f"Synthetic demo plant: 8 HV motors (pumps, fans, compressor, crusher), status on {DEMO_DATE}. Not real measurements.")
    if not R0:
        st.markdown("<h3 style='text-align:center;color:#888;margin-top:30px'>💡 No results yet: analyze a motor on the "
                    "Motor Risk & FMEA page, or show the demo plant.</h3>", unsafe_allow_html=True)
    else:
        R = pd.DataFrame(R0).sort_values('Overall Risk (%)', ascending=False).reset_index(drop=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Motors", len(R))
        k2.metric("🔴 Action required", int((R['Status'] == 'ACTION').sum()))
        k3.metric("🟠 Watch", int((R['Status'] == 'WATCH').sum()))
        k4.metric("🟢 Normal", int((R['Status'] == 'EXCELLENT').sum()))
        w = R.iloc[0]
        rp = R.sort_values('RPN', ascending=False).iloc[0]
        zd = R.dropna(subset=['Days to Zone D'])
        zd = zd[zd['Days to Zone D'] > 0].sort_values('Days to Zone D')
        hot = R.sort_values('Stator Temp (C)', ascending=False).iloc[0]
        cards = [("Highest risk", f"{w['Overall Risk (%)']:.0f}%", STATUS_COLOR[w['Status']], f"{w['Motor ID']} · {w['Governing Mode']}"),
                 ("Highest RPN", f"{int(rp['RPN'])}", GREEN if rp['RPN'] < 40 else AMBER if rp['RPN'] < 100 else RED,
                  f"{rp['Motor ID']} · {rp['RPN Level']}"),
                 ("Shortest time to vibration zone D", f"{int(zd.iloc[0]['Days to Zone D'])} d" if len(zd) else "—",
                  AMBER if len(zd) else '#999', zd.iloc[0]['Motor ID'] if len(zd) else "no rising trend"),
                 ("Hottest stator", f"{hot['Stator Temp (C)']:.0f} °C", NAVY, hot['Motor ID'])]
        for col, (t, big, c, sub) in zip(st.columns(4), cards):
            col.markdown(f"<div class='pg-card' style='border-top:5px solid {c}'><h4>{t}</h4><div class='big' style='color:{c}'>{big}</div>"
                         f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([1.1, 1])
        with left:
            st.plotly_chart(ranking_plot(R, top=15), width=STRETCH, key='ov_rank')
        with right:
            st.plotly_chart(gov_plot(R), width=STRETCH, key='ov_gov')
        st.plotly_chart(heat_plot(R), width=STRETCH, key='ov_heat')
        st.markdown("#### All motors")
        show_cols = ['Motor ID', 'Service', 'Governing Mode', 'Overall Risk (%)', 'Status', 'RPN', 'RPN Level', 'Load (%)',
                     'Stator Temp (C)', 'Vibration Max (mm/s)', 'ISO Zone', 'Days to Zone D', 'Source', 'Assessed', 'Action']
        st.dataframe(R[show_cols].round(1), hide_index=True, width=STRETCH)
        st.download_button("📥 Download motor ranking (CSV)", R[show_cols].round(1).to_csv(index=False).encode('utf-8-sig'),
                           f"MotorGuard_overview_{datetime.now():%Y%m%d}.csv", 'text/csv')

# ---------------- Analysis pages
with tabs[1]:
    motor_page()
with tabs[2]:
    check_page('th', 'thermal')
with tabs[3]:
    check_page('me', 'mech')
with tabs[4]:
    check_page('su', 'supply')

# ---------------- Combined report
with tabs[5]:
    recs = current_records()
    if not recs:
        st.markdown("<h3 style='text-align:center;color:#888;margin-top:40px'>💡 Analyze motors on the Motor Risk & FMEA page "
                    "(or show the demo plant on the Overview page) first.</h3>", unsafe_allow_html=True)
    else:
        R = pd.DataFrame(recs).sort_values('Overall Risk (%)', ascending=False).reset_index(drop=True)
        n_a, n_w = int((R['Status'] == 'ACTION').sum()), int((R['Status'] == 'WATCH').sum())
        st.markdown(f"<div class='pg-note'><b>📄 Combined Motor Condition Report</b><br>{len(R)} motors · source: "
                    f"{st.session_state.get('ov_src', SRC_PAGES).lower()} · generated {datetime.now():%Y-%m-%d %H:%M}</div>",
                    unsafe_allow_html=True)
        top = R.iloc[0]
        if n_a:
            summary = (f"{n_a} motor(s) need action and {n_w} are on watch. Highest risk: {top['Motor ID']} "
                       f"({top['Governing Mode']}, {top['Overall Risk (%)']:.0f}%). {top['Action']}")
        elif n_w:
            summary = f"No motor needs immediate action; {n_w} motor(s) are on watch. Highest: {top['Motor ID']} ({top['Governing Mode']})."
        else:
            summary = "All assessed motors are within normal limits."
        st.markdown(f"**Executive summary:** {summary}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Motors", len(R)); k2.metric("🔴 Action", n_a); k3.metric("🟠 Watch", n_w); k4.metric("🟢 Normal", len(R) - n_a - n_w)
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.plotly_chart(ranking_plot(R, top=15), width=STRETCH, key='rep_rank')
        with c2:
            st.plotly_chart(gov_plot(R), width=STRETCH, key='rep_gov')
        cols = ['Motor ID', 'Service', 'Governing Mode', 'Overall Risk (%)', 'Status', 'RPN', 'RPN Level', 'Days to Zone D', 'Source', 'Assessed', 'Action']
        st.markdown("#### All motors")
        st.dataframe(R[cols].round(1), hide_index=True, width=STRETCH)
        st.markdown("#### Motors above the normal band")
        above = R[R['Status'] != 'EXCELLENT']
        if above.empty:
            st.success("No motor above the normal band.")
        for i, (_, r) in enumerate(above.iterrows()):
            with st.expander(f"{r['Motor ID']}  ·  {r['Governing Mode']}  ·  {r['Overall Risk (%)']:.0f}% {r['Status']}", expanded=i < 3):
                e1, e2 = st.columns([1, 1.1])
                with e1:
                    for t in r['Findings']:
                        st.markdown(f"- {t}")
                    st.markdown(f"**RPN** = {r['S']} × {r['O']} × {r['D']} = **{r['RPN']}** ({r['RPN Level']})")
                with e2:
                    st.plotly_chart(mode_plot(r['Mode Risks'], height=280), width=STRETCH, key=f"rep_m_{i}")
                st.warning(f"**Action:** {r['Action']}")
        lines = ["=========================================", "MOTORGUARD AI - COMBINED MOTOR CONDITION REPORT",
                 "=========================================", f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
                 f"Motors: {len(R)}  (action {n_a}, watch {n_w})", "", "EXECUTIVE SUMMARY:", summary, ""]
        for _, r in R.iterrows():
            lines += ["-----------------------------------------",
                      f"{r['Motor ID']} ({r['Service']}) - {r['Status']} - failure risk {r['Overall Risk (%)']:.0f}%",
                      f"Governing mode: {r['Governing Mode']}  |  RPN = {r['S']} x {r['O']} x {r['D']} = {r['RPN']} ({r['RPN Level']})",
                      "Mode risks: " + ", ".join(f"{m} {v:.0f}%" for m, v in r['Mode Risks'].items()),
                      f"Source: {r['Source']} ({r['Assessed']})"] + [f"* {t}" for t in r['Findings']] + [f"Action: {r['Action']}"]
        lines += ["=========================================", "Generated by MotorGuard AI Engine"]
        fm = pd.DataFrame([dict(**{'Motor ID': r['Motor ID']}, **m) for _, r in R.iterrows() for m in r['FMEA']])
        d1, d2, d3 = st.columns(3)
        d1.download_button("📥 Download Full Text Report", "\n".join(lines).encode('utf-8'), f"MotorGuard_Report_{datetime.now():%Y%m%d}.txt", 'text/plain')
        d2.download_button("📥 Download Table (CSV)", R[cols].round(1).to_csv(index=False).encode('utf-8-sig'), f"MotorGuard_Report_{datetime.now():%Y%m%d}.csv", 'text/csv')
        d3.download_button("📥 Download FMEA, all modes (CSV)", fm.to_csv(index=False).encode('utf-8-sig'), f"MotorGuard_FMEA_{datetime.now():%Y%m%d}.csv", 'text/csv')

# ---------------- Historical study
with tabs[6]:
    st.markdown("<div class='pg-note'>Replays every reading in date order. Each assessment uses only the data that "
                "existed on that date (7-day levels, 60-day trends, the latest test results), then the alerts are compared "
                "with the recorded failures and findings. The result shows how early the system would have warned.</div>",
                unsafe_allow_html=True)
    hsrc = st.radio("Data", ["Demo plant (synthetic, Jan 2025 - Aug 2026)", "Upload client files"], horizontal=True, key='h_src')
    hfiles, herr = (None, [])
    if hsrc.startswith("Upload"):
        hfiles, herr = read_uploads('h')
        with st.expander("📄 File formats and demo files"):
            st.markdown("""
- **Motor register:** `Motor ID, Service, Rated kW, Rated Current (A), Rated Voltage (kV), Insulation Class (B/F/H), Rise Class (B/F/H),
  Bearing Type (Rolling/Sleeve), Foundation (Rigid/Flexible), Allowed Starts per Day, Rated Start Time (s)`.
- **Online readings** (one row per motor per day or per hour): `Timestamp, Motor ID, Ia, Ib, Ic (A), Vab, Vbc, Vca (kV), Ambient (C),
  Stator Temp (C), Bearing DE Temp (C), Bearing NDE Temp (C), Vibration DE (mm/s), Vibration NDE (mm/s), Starts, Max Start Time (s)`.
- **Periodic tests:** `Date, Motor ID, IR 1min (MOhm), PI, MCSA Sideband (dB)` (any column may be empty).
- **Events:** `Motor ID, Date, Event`.
""")
            demo = demo_full()
            dc = st.columns(4)
            for col, kk in zip(dc, NEED):
                col.download_button(kk.capitalize(), demo[kk].to_csv(index=False).encode('utf-8-sig'), f"demo_motor_{kk}.csv",
                                    'text/csv', key=f"dl_{kk}")
    h1, h2 = st.columns(2)
    thr = h1.number_input("Alert threshold (%)", 50, 95, 70, 5)
    lb = h2.number_input("Look-back window (days)", 30, 1095, 365, 30)
    if st.button("▶️ Run Historical Study", type='primary', key='h_go'):
        data = None
        if hsrc.startswith("Demo"):
            data = demo_full()
            rows = demo_series()
        else:
            for e in herr:
                st.error(e)
            if not herr and hfiles['register'] is not None and hfiles['online'] is not None:
                data = hfiles
                rows = run_series(data['online'], data['register'], data.get('tests'))
            elif not herr:
                st.error("Load at least the motor register and the online readings, plus the events file.")
        if data is not None:
            if not rows:
                st.error("No readings match the motor register (check the Motor ID column).")
            else:
                tl, evs, fa, n_ep = mc.historical_study(rows, data.get('events'), thr, lb)
                modes = pd.DataFrame([dict(**{'Motor ID': a['Motor ID'], 'Date': pd.Timestamp(a['Timestamp']).floor('D')}, **a['Mode Risks'])
                                      for a in rows]).groupby(['Motor ID', 'Date'], as_index=False).last()
                st.session_state['hist'] = dict(tl=tl, evs=evs, fa=fa, n_ep=n_ep, thr=thr, events=data.get('events'),
                                                modes=modes, demo=hsrc.startswith("Demo"))
    H = st.session_state['hist']
    if H is not None:
        tl, evs = H['tl'], H['evs']
        if H['demo']:
            st.info("Synthetic demo data. Not real measurements.")
        caught = int((evs['Caught'] == 'Yes').sum()) if len(evs) else 0
        lt = evs['Lead Time (days)'].dropna() if len(evs) else pd.Series(dtype=float)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Motors", tl['Motor ID'].nunique())
        k2.metric("Events caught", f"{caught} / {len(evs)}" if len(evs) else "no events")
        k3.metric("Median warning before event", f"{int(lt.median())} days" if len(lt) else "—")
        k4.metric("Alert episodes with no event", f"{H['fa']} of {H['n_ep']}")
        show = tl.groupby('Motor ID')['Overall Risk (%)'].max()
        show = show[show > mc.EXCELLENT_MAX].index.tolist()
        fig = go.Figure()
        pal = ['#2a78d6', '#eb6834', '#1baf7a', '#8e5cd6', '#e87ba4', '#555555', '#eda100', '#008300']
        for i, mid in enumerate(show):
            g = tl[tl['Motor ID'] == mid]
            fig.add_scatter(x=g['Date'], y=g['Overall Risk (%)'], mode='lines', name=mid, line=dict(color=pal[i % len(pal)]))
        fig.add_hline(y=H['thr'], line_dash='dash', line_color=RED, annotation_text=f"Alert threshold {H['thr']}%", annotation_position='top left')
        if H['events'] is not None:
            for _, e in H['events'].iterrows():
                d_ = pd.to_datetime(e['Date'], errors='coerce')
                if pd.notna(d_):
                    fig.add_vline(x=d_, line_dash='dot', line_color='#555')
                    fig.add_annotation(x=d_, y=104, text=f"⚠ {e['Motor ID']}", showarrow=False)
        fig.update_layout(title='Failure risk over time (motors that ever left the normal band)', height=440,
                          yaxis=dict(range=[0, 110], title='Failure Risk (%)'), legend=dict(orientation='h', y=-0.2),
                          margin=dict(l=50, r=20, t=50, b=40))
        st.plotly_chart(fig, width=STRETCH, key='h_fig')
        if len(evs):
            st.markdown("#### Events vs. warnings")
            st.dataframe(evs, hide_index=True, width=STRETCH)
        st.markdown("#### One motor in detail")
        mids = sorted(tl['Motor ID'].unique())
        sel = st.selectbox("Motor", mids, index=mids.index(show[0]) if show else 0, key='h_sel')
        g = H['modes'][H['modes']['Motor ID'] == sel]
        fig2 = go.Figure()
        for i, m in enumerate(MODE_ORDER):
            if g[m].max() > 5:
                fig2.add_scatter(x=g['Date'], y=g[m], mode='lines', name=m, line=dict(color=pal[i % len(pal)]))
        fig2.add_hline(y=35, line_dash='dash', line_color=AMBER); fig2.add_hline(y=70, line_dash='dash', line_color=RED)
        if H['events'] is not None:
            for _, e in H['events'][H['events']['Motor ID'].astype(str) == sel].iterrows():
                fig2.add_vline(x=pd.to_datetime(e['Date']), line_dash='dot', line_color='#555')
                fig2.add_annotation(x=pd.to_datetime(e['Date']), y=104, text=f"⚠ {str(e['Event'])[:40]}", showarrow=False)
        fig2.update_layout(title=f'{sel}: risk by failure mode over time', height=380, yaxis=dict(range=[0, 110], title='Risk (%)'),
                           legend=dict(orientation='h', y=-0.2), margin=dict(l=50, r=20, t=50, b=40))
        st.plotly_chart(fig2, width=STRETCH, key='h_fig2')
        st.caption("'Alert episodes with no event' counts alert periods (alerts less than 120 days apart form one episode) "
                   "not followed by a recorded event within the look-back window.")
        st.download_button("📥 Download daily motor timeline (CSV)", tl.to_csv(index=False).encode('utf-8-sig'),
                           'motorguard_historical_timeline.csv', 'text/csv')
    with st.expander("How the failure risk is calculated"):
        st.markdown(f"""
Each motor gets a risk (0-100%) for seven failure modes:

| Failure mode | Main inputs | Standard / rule |
|---|---|---|
| Stator insulation | hottest stator RTD, IR and PI tests | class limit B 130 / F 155 / H 180 °C; alarm = limit − 25 °C, trip = limit − 15 °C; ageing doubles every 10 °C; IEEE 43 (PI ≥ 2.0, IR ≥ 100 MΩ) |
| Cooling | stator temperature against the **expected** temperature | machine-learning thermal model (load, current unbalance, ambient); ≥ {mc.COOLING_DEV_WATCH:.0f} °C above expected = watch |
| Bearing | DE / NDE temperatures, their difference and trend | rolling 90 / 100 °C, sleeve 85 / 95 °C (alarm / trip); DE − NDE ≥ {mc.BEARING_DIFF:.0f} °C |
| Mechanical vibration | DE / NDE velocity and its trend | ISO 20816-3 zones A-D by machine group and foundation; days to zone D from the 60-day trend |
| Supply | phase voltages and currents | NEMA MG1 voltage unbalance and derating; current unbalance; overload |
| Rotor | start time, MCSA sideband | start time ≥ {mc.START_RATIO_WATCH}× rated rises the risk; sideband ≥ {mc.MCSA_GOOD:.0f} dB healthy, ≤ {mc.MCSA_BAD:.0f} dB broken bars |
| Starting duty | starts per day | against the allowed number |

**Overall risk** = the highest mode risk + 5 points for every other mode above normal. Bands: ≤ 35% normal, ≤ 70% watch, > 70% action.
**FMEA:** RPN = S × O × D with S per failure mode, O = risk ÷ 10, D = 2 for online monitoring.

The thermal model is trained on physics-based synthetic data (I²R losses, negative-sequence heating). It must be
re-trained on each motor's own healthy history before any accuracy figure is quoted.
""")

st.markdown(FOOTER, unsafe_allow_html=True)
