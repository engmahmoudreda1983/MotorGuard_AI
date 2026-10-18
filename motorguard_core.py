"""
MotorGuard AI - condition and risk engine for HV induction motors.

One engine used by every page of the app, the batch analysis and the historical study.

Inputs (CSV):
  * Motor register : Motor ID, Service, Rated kW, Rated Current (A), Rated Voltage (kV), Insulation Class,
                     Rise Class, Bearing Type, Foundation, Allowed Starts per Day, Rated Start Time (s)
  * Online readings: Timestamp, Motor ID, Ia, Ib, Ic (A), Vab, Vbc, Vca (kV), Ambient (C), Stator Temp (C),
                     Bearing DE Temp (C), Bearing NDE Temp (C), Vibration DE (mm/s), Vibration NDE (mm/s),
                     Starts, Max Start Time (s)
  * Periodic tests (optional): Date, Motor ID, IR 1min (MOhm), PI, MCSA Sideband (dB)
  * Events (for the historical study): Motor ID, Date, Event
"""
import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- standards and limits
CLASS_LIMIT = {'B': 130.0, 'F': 155.0, 'H': 180.0}          # IEC 60034-1 / NEMA MG1 hot-spot class limits
RISE_RATED = {'B': 80.0, 'F': 105.0, 'H': 125.0}            # rated temperature rise by resistance (K)
STATOR_ALARM_BELOW = 25.0    # alarm = class limit - 25 C  (class F: 130 C)
STATOR_TRIP_BELOW = 15.0     # trip  = class limit - 15 C  (class F: 140 C)
BEARING_LIMITS = {'Rolling': (90.0, 100.0), 'Sleeve': (85.0, 95.0)}   # alarm, trip (C)
BEARING_DIFF = 15.0          # DE-NDE difference that flags one bearing
COOLING_DEV_WATCH = 10.0     # measured stator temp above expected (C)
COOLING_DEV_FULL = 25.0
# ISO 20816-3 zone boundaries, velocity r.m.s. mm/s: (A/B, B/C, C/D)
ISO_ZONES = {('Group 1', 'Rigid'): (2.3, 4.5, 7.1), ('Group 1', 'Flexible'): (3.5, 7.1, 11.0),
             ('Group 2', 'Rigid'): (1.4, 2.8, 4.5), ('Group 2', 'Flexible'): (2.3, 4.5, 7.1)}
# NEMA MG1 derating for voltage unbalance (%, factor)
NEMA_DERATE = [(0, 1.0), (1, 0.98), (2, 0.95), (3, 0.88), (4, 0.82), (5, 0.75)]
# MCSA rotor-bar sideband (dB below supply): higher is better
MCSA_GOOD, MCSA_BAD = 54.0, 36.0
PI_MIN, PI_GOOD = 1.5, 2.0   # IEEE 43: minimum recommended PI 2.0 for class B and above
IR_MIN = 100.0               # IEEE 43: 100 MOhm for form-wound stators (1 min, 40 C)
START_RATIO_WATCH, START_RATIO_FULL = 1.15, 1.6
TREND_WINDOW_D, LEVEL_WINDOW_D, TEST_VALID_D, START_VALID_D = 60, 7, 400, 30
EXCELLENT_MAX, WATCH_MAX = 35.0, 70.0
DETECTION_ONLINE = 2

MODES = {
    'Stator insulation': {'S': 9, 'text': 'Stator winding insulation (thermal ageing, overheating, insulation tests)'},
    'Bearing': {'S': 8, 'text': 'Bearing damage (lubrication, wear, shaft currents)'},
    'Rotor': {'S': 8, 'text': 'Rotor cage (broken or cracked bars)'},
    'Mechanical vibration': {'S': 7, 'text': 'Unbalance, misalignment or looseness'},
    'Cooling': {'S': 7, 'text': 'Cooling deficiency (blocked air path, fan, cooler)'},
    'Supply': {'S': 6, 'text': 'Supply problem (voltage unbalance, single-phasing risk, overload)'},
    'Starting duty': {'S': 5, 'text': 'Excessive starting duty'},
}
ACTIONS = {
    'Stator insulation': 'Reduce load if possible; plan insulation tests (IR/PI, tan delta, PD) and a stator inspection at the next outage.',
    'Bearing': 'Check lubrication (grease quantity and condition or oil level); plan a bearing inspection or replacement; check shaft grounding.',
    'Rotor': 'Confirm with a motor current signature test; plan a rotor inspection; limit the number of starts.',
    'Mechanical vibration': 'Check alignment, balance and foundation bolts; take a vibration spectrum to find the cause.',
    'Cooling': 'Inspect the cooling path: air filters, cooler tubes, fan and ventilation openings.',
    'Supply': 'Check supply voltages, cable joints and breaker contacts; reduce load until the unbalance is corrected.',
    'Starting duty': 'Reduce starts per day; respect the cold/hot start limits; check the driven-load start sequence.',
}


def clamp(x, lo=0.0, hi=100.0):
    return float(min(hi, max(lo, x)))


def lin(x, x0, x1, y0=0.0, y1=100.0):
    """Linear score from y0 at x0 to y1 at x1 (clamped)."""
    if x1 == x0:
        return y1 if x >= x1 else y0
    return clamp(y0 + (y1 - y0) * (x - x0) / (x1 - x0), min(y0, y1), max(y0, y1))


def status_of(r):
    return 'EXCELLENT' if r <= EXCELLENT_MAX else ('WATCH' if r <= WATCH_MAX else 'ACTION')


def rpn_level(rpn):
    return 'Low' if rpn < 40 else ('Medium' if rpn < 100 else 'High')


def unbalance(a, b, c):
    """NEMA definition: max deviation from the average, % of the average."""
    v = [float(a), float(b), float(c)]
    m = sum(v) / 3
    return 0.0 if m <= 0 else max(abs(x - m) for x in v) / m * 100


def nema_derate(vu):
    if vu >= 5:
        return 0.75
    for (x0, y0), (x1, y1) in zip(NEMA_DERATE, NEMA_DERATE[1:]):
        if vu <= x1:
            return y0 + (y1 - y0) * (vu - x0) / (x1 - x0)
    return 0.75


def iso_group(rated_kw):
    return 'Group 1' if float(rated_kw) > 300 else 'Group 2'


def iso_bounds(rated_kw, foundation):
    f = 'Flexible' if str(foundation).strip().lower().startswith('flex') else 'Rigid'
    return ISO_ZONES[(iso_group(rated_kw), f)]


def iso_zone(v, bounds):
    ab, bc, cd = bounds
    return 'A' if v < ab else 'B' if v < bc else 'C' if v < cd else 'D'


def vib_score(v, bounds):
    ab, bc, cd = bounds
    if v < ab:
        return lin(v, 0, ab, 0, 10)
    if v < bc:
        return lin(v, ab, bc, 10, 35)
    if v < cd:
        return lin(v, bc, cd, 35, 70)
    return lin(v, cd, cd * 1.5, 70, 100)


# ---------------------------------------------------------------- thermal model
THERMAL_FEATURES = ['Load (%)', 'Current Unbalance (%)', 'Ambient (C)']


def thermal_physics(load, iunb, amb, rng=None):
    """Stator rise as a fraction of the rated rise: no-load share + I^2 R losses, extra loss from
    negative-sequence current, small loss of cooling with hot air."""
    frac = 0.12 + 0.88 * (load / 100.0) ** 2 * (1 + 4.0 * (iunb / 100.0) ** 2 * 6) * (1 + 0.004 * (amb - 40))
    if rng is not None:
        frac = frac * (1 + rng.normal(0, 0.03, np.shape(load)))
    return frac


def make_thermal_training(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    load = rng.uniform(5, 125, n); iunb = np.abs(rng.normal(0, 4, n)); amb = rng.uniform(10, 55, n)
    X = np.c_[load, iunb, amb]
    return X, thermal_physics(load, iunb, amb, rng)


def expected_stator_temp(model, load, iunb, amb, rise_class):
    frac = float(model.predict(np.array([[load, iunb, amb]]))[0])
    return amb + frac * RISE_RATED.get(str(rise_class).upper(), 105.0)


# ---------------------------------------------------------------- one reading
def reg_value(reg, key, default):
    v = reg.get(key, default)
    try:
        return default if v is None or (isinstance(v, float) and math.isnan(v)) or v == '' else v
    except Exception:
        return default


def assess(reading, reg, model, tests=None, trend=None, exp_frac=None, excess=None):
    """Assess one motor reading with the same rules everywhere.
    reading: dict with Ia, Ib, Ic, Vab, Vbc, Vca, Ambient (C), Stator Temp (C), Bearing DE/NDE Temp (C),
             Vibration DE/NDE (mm/s), Starts, Max Start Time (s)
    reg: dict with the motor register fields
    tests: dict with IR 1min (MOhm), PI, MCSA Sideband (dB) (any may be None)
    trend: dict with Vibration Trend (mm/s/30d), Bearing Temp Trend (C/30d) (optional)"""
    tests = tests or {}; trend = trend or {}
    g = lambda k, d=0.0: float(reading.get(k, d) if reading.get(k, d) not in (None, '') and not
                               (isinstance(reading.get(k), float) and math.isnan(reading.get(k))) else d)
    kw = float(reg_value(reg, 'Rated kW', 1000)); irated = float(reg_value(reg, 'Rated Current (A)', 100))
    ins = str(reg_value(reg, 'Insulation Class', 'F')).upper()[:1]; rise = str(reg_value(reg, 'Rise Class', 'B')).upper()[:1]
    btype = 'Sleeve' if str(reg_value(reg, 'Bearing Type', 'Rolling')).lower().startswith('sl') else 'Rolling'
    found = reg_value(reg, 'Foundation', 'Rigid')
    allowed = float(reg_value(reg, 'Allowed Starts per Day', 4)); tstart_r = float(reg_value(reg, 'Rated Start Time (s)', 10))

    ia, ib, ic = g('Ia'), g('Ib'), g('Ic')
    vab, vbc, vca = g('Vab'), g('Vbc'), g('Vca')
    amb, ts = g('Ambient (C)', 35), g('Stator Temp (C)')
    bde, bnde = g('Bearing DE Temp (C)'), g('Bearing NDE Temp (C)')
    vde, vnde = g('Vibration DE (mm/s)'), g('Vibration NDE (mm/s)')
    starts, tst = g('Starts'), g('Max Start Time (s)')
    iavg = (ia + ib + ic) / 3
    load = iavg / irated * 100 if irated > 0 else 0
    iu, vu = unbalance(ia, ib, ic), unbalance(vab, vbc, vca)
    derate = nema_derate(vu)
    load_eff = load / derate if derate > 0 else load
    running = load >= 5

    out = {'Motor ID': reg_value(reg, 'Motor ID', reading.get('Motor ID', '')), 'Service': reg_value(reg, 'Service', ''),
           'Rated kW': kw, 'Load (%)': load, 'Effective Load (%)': load_eff, 'Current Unbalance (%)': iu,
           'Voltage Unbalance (%)': vu, 'NEMA Derating': derate, 'Ambient (C)': amb, 'Stator Temp (C)': ts,
           'Bearing DE Temp (C)': bde, 'Bearing NDE Temp (C)': bnde, 'Vibration DE (mm/s)': vde, 'Vibration NDE (mm/s)': vnde,
           'Starts': starts, 'Max Start Time (s)': tst, 'Running': running}
    R, why = {}, {}

    # --- stator insulation
    lim = CLASS_LIMIT.get(ins, 155.0); alarm, trip = lim - STATOR_ALARM_BELOW, lim - STATOR_TRIP_BELOW
    if ts < alarm:
        s_t = lin(ts, alarm - 30, alarm, 0, 50)
    elif ts < trip:
        s_t = lin(ts, alarm, trip, 50, 85)
    else:
        s_t = lin(ts, trip, lim, 85, 100)
    aging = 2 ** ((ts - (lim - 20)) / 10.0) if running else 0.0   # 1.0 at 20 C below the class limit
    out.update({'Stator Alarm (C)': alarm, 'Stator Trip (C)': trip, 'Class Limit (C)': lim, 'Ageing Factor': aging})
    w = [f"Stator {ts:.0f} °C (class {ins}: alarm {alarm:.0f} °C, trip {trip:.0f} °C); insulation ageing {aging:.2f}× normal."]
    pi, ir = tests.get('PI'), tests.get('IR 1min (MOhm)')
    if pi is not None and not pd.isna(pi):
        if pi < PI_MIN:
            s_t = max(s_t, 75); w.append(f"Polarization index {pi:.1f} is below 1.5: insulation contaminated or damaged (IEEE 43).")
        elif pi < PI_GOOD:
            s_t = max(s_t, 45); w.append(f"Polarization index {pi:.1f} is below the IEEE 43 minimum of 2.0.")
        else:
            w.append(f"Polarization index {pi:.1f} is acceptable.")
    if ir is not None and not pd.isna(ir):
        if ir < IR_MIN:
            s_t = max(s_t, 70); w.append(f"Insulation resistance {ir:.0f} MΩ is below the IEEE 43 minimum of {IR_MIN:.0f} MΩ.")
    if aging >= 4:
        s_t = max(s_t, 60)
    R['Stator insulation'], why['Stator insulation'] = s_t, w

    # --- cooling (measured vs expected stator temperature)
    if exp_frac is not None and running:
        exp_t = amb + exp_frac * RISE_RATED.get(rise, 105.0)
    else:
        exp_t = expected_stator_temp(model, load_eff, iu, amb, rise) if (model is not None and running) else None
    if excess is not None and exp_t is not None:
        exp_t = ts - excess                 # smoothed excess from the series (7-day median of daily excess)
    dev = ts - exp_t if exp_t is not None else 0.0
    out['Expected Stator Temp (C)'], out['Stator Temp Excess (C)'] = exp_t, dev if exp_t is not None else None
    s_c = lin(dev, 5, COOLING_DEV_FULL) if exp_t is not None else 0.0
    if exp_t is not None and dev >= COOLING_DEV_WATCH:
        s_c = max(s_c, 40)
    R['Cooling'] = s_c
    why['Cooling'] = [f"Stator {ts:.0f} °C against {exp_t:.0f} °C expected at this load and ambient ({dev:+.0f} °C)."] if exp_t is not None else ['Motor stopped: no comparison.']

    # --- bearing
    ba, bt = BEARING_LIMITS[btype]
    bmax = max(bde, bnde)
    s_b = lin(bmax, ba - 25, ba, 0, 50) if bmax < ba else lin(bmax, ba, bt, 50, 85) if bmax < bt else lin(bmax, bt, bt + 10, 85, 100)
    w = [f"Bearings DE {bde:.0f} °C / NDE {bnde:.0f} °C ({btype.lower()} bearings: alarm {ba:.0f} °C, trip {bt:.0f} °C)."]
    diff = abs(bde - bnde)
    if running and diff >= BEARING_DIFF:
        s_b += 15; w.append(f"{'DE' if bde > bnde else 'NDE'} bearing runs {diff:.0f} °C hotter than the other one.")
    bt_tr = trend.get('Bearing Temp Trend (C/30d)')
    if bt_tr is not None and bt_tr > 2:
        s_b += lin(bt_tr, 2, 8, 5, 20); w.append(f"Bearing temperature rising {bt_tr:.1f} °C per month.")
    bounds = iso_bounds(kw, found)
    vmax = max(vde, vnde); zone = iso_zone(vmax, bounds)
    if zone in ('C', 'D') and running:
        s_b += 10 if zone == 'C' else 20
    R['Bearing'], why['Bearing'] = clamp(s_b), w

    # --- mechanical vibration (ISO 20816-3)
    s_v = vib_score(vmax, bounds) if running else 0.0
    out.update({'ISO Group': iso_group(kw), 'Zone Bounds': bounds, 'Vibration Max (mm/s)': vmax, 'ISO Zone': zone})
    w = [f"Vibration {vmax:.1f} mm/s r.m.s.: ISO 20816-3 zone {zone} ({iso_group(kw).lower()}, {str(found).lower()} foundation; C/D boundary {bounds[2]} mm/s)."]
    v_tr = trend.get('Vibration Trend (mm/s/30d)')
    ttd = None
    if v_tr is not None and running:
        if v_tr > 0.1:
            s_v += lin(v_tr / bounds[2] * 100, 2, 15, 0, 20)
            w.append(f"Rising {v_tr:.2f} mm/s per month.")
            if vmax < bounds[2]:
                ttd = (bounds[2] - vmax) / v_tr * 30
                w.append(f"At this rate it reaches zone D in about {ttd:.0f} days.")
    if zone == 'D' and running:
        ttd = 0
    out['Days to Zone D'] = None if ttd is None else round(ttd)
    R['Mechanical vibration'], why['Mechanical vibration'] = clamp(s_v), w

    # --- supply
    s_s = max(lin(vu, 1, 5), lin(iu, 5, 20), lin(load_eff, 100, 120, 0, 100) if load_eff > 100 else 0.0)
    w = [f"Voltage unbalance {vu:.1f}% (NEMA MG1 derating ×{derate:.2f}); current unbalance {iu:.1f}%; load {load:.0f}% of rated, {load_eff:.0f}% after derating."]
    R['Supply'], why['Supply'] = s_s, w

    # --- rotor
    ratio = tst / tstart_r if (tst > 0 and tstart_r > 0) else None
    s_r = lin(ratio, START_RATIO_WATCH, START_RATIO_FULL) if ratio else 0.0
    w = [f"Last start took {tst:.1f} s ({ratio:.2f}× the rated {tstart_r:.0f} s)."] if ratio else ['No start in the last 30 days.']
    sb = tests.get('MCSA Sideband (dB)')
    if sb is not None and not pd.isna(sb):
        s_mc = lin(sb, MCSA_GOOD, MCSA_BAD)
        s_r = max(s_r, s_mc)
        w.append(f"Rotor-bar sideband {sb:.0f} dB below supply ({'healthy' if sb >= 48 else 'cracked bars likely' if sb >= 42 else 'broken bars likely'}).")
        if ratio and ratio >= START_RATIO_WATCH and sb < 48:
            s_r = max(s_r, 75); w.append('Long start time and MCSA sideband agree: rotor defect confirmed by two methods.')
    R['Rotor'], why['Rotor'] = s_r, w

    # --- starting duty
    s_d = lin(starts / allowed if allowed > 0 else 0, 0.8, 2.0)
    R['Starting duty'], why['Starting duty'] = s_d, [f"{starts:.0f} starts against {allowed:.0f} allowed per day."]

    # --- overall
    risks = {k: round(v, 1) for k, v in R.items()}
    gov = max(risks, key=risks.get)
    others = sum(1 for k, v in risks.items() if k != gov and v > EXCELLENT_MAX)
    overall = clamp(risks[gov] + 5 * others)
    S = MODES[gov]['S']; O = max(1, math.ceil(overall / 10)); rpn = S * O * DETECTION_ONLINE
    modes = []
    for k, v in sorted(risks.items(), key=lambda kv: -kv[1]):
        o = max(1, math.ceil(v / 10))
        modes.append({'Failure Mode': k, 'Risk (%)': v, 'S': MODES[k]['S'], 'O': o, 'D': DETECTION_ONLINE,
                      'RPN': MODES[k]['S'] * o * DETECTION_ONLINE, 'RPN Level': rpn_level(MODES[k]['S'] * o * DETECTION_ONLINE)})
    out.update({'Mode Risks': risks, 'Reasons': why, 'Governing Mode': gov, 'Other Modes Above Normal': others,
                'Overall Risk (%)': round(overall, 1), 'Status': status_of(round(overall, 1)), 'S': S, 'O': O, 'D': DETECTION_ONLINE,
                'RPN': rpn, 'RPN Level': rpn_level(rpn), 'FMEA': modes,
                'Action': ACTIONS[gov] if status_of(round(overall, 1)) != 'EXCELLENT' else 'Continue online monitoring.'})
    return out


def findings(a):
    """Plain-language findings: governing mode first, then the other modes above normal."""
    r = []
    order = sorted(a['Mode Risks'].items(), key=lambda kv: -kv[1])
    for k, v in order:
        if k == a['Governing Mode'] or v > EXCELLENT_MAX:
            r.append(f"{k} ({v:.0f}%): " + ' '.join(a['Reasons'][k]))
    if a['Other Modes Above Normal']:
        r.append(f"{a['Other Modes Above Normal']} other failure mode(s) above normal add {5 * a['Other Modes Above Normal']} points to the overall risk.")
    if a['Status'] == 'EXCELLENT':
        r.append('All failure modes within normal limits.')
    return r


# ---------------------------------------------------------------- time series
def _rolling(ts, vals, days, how):
    s = pd.Series(vals, index=pd.DatetimeIndex(ts))
    r = s.rolling(f"{days}D")
    if how == 'median':
        return r.median().to_numpy()
    t = pd.Series((s.index - s.index[0]).days.astype(float), index=s.index)
    n = r.count()
    mt, my = t.rolling(f"{days}D").mean(), r.mean()
    cov = (t * s).rolling(f"{days}D").mean() - mt * my
    var = (t * t).rolling(f"{days}D").mean() - mt * mt
    sl = (cov / var.replace(0, np.nan)) * 30
    sl[n < 10] = 0.0
    return sl.fillna(0).to_numpy()


SMOOTHED = ['Ambient (C)', 'Stator Temp (C)', 'Bearing DE Temp (C)', 'Bearing NDE Temp (C)', 'Vibration DE (mm/s)', 'Vibration NDE (mm/s)']


def assess_series(online, register, model, tests=None):
    """Assess every reading using only the data that existed on that date."""
    df = online.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    reg = {str(r['Motor ID']): r for r in register.to_dict('records')}
    df['Motor ID'] = df['Motor ID'].astype(str)
    df = df[df['Motor ID'].isin(reg)].sort_values(['Motor ID', 'Timestamp']).reset_index(drop=True)
    num = ['Ia', 'Ib', 'Ic', 'Vab', 'Vbc', 'Vca', 'Ambient (C)', 'Stator Temp (C)', 'Bearing DE Temp (C)', 'Bearing NDE Temp (C)',
           'Vibration DE (mm/s)', 'Vibration NDE (mm/s)', 'Starts', 'Max Start Time (s)']
    for c in num:
        df[c] = pd.to_numeric(df.get(c), errors='coerce')
    df[['Starts', 'Max Start Time (s)']] = df[['Starts', 'Max Start Time (s)']].fillna(0)
    # the last start in the previous 30 days stays valid on days without a start (rotor check)
    df['_tst'] = df['Max Start Time (s)']
    for mid, g in df.groupby('Motor ID'):
        st_ = g[(g['Starts'] > 0) & (g['Max Start Time (s)'] > 0)][['Timestamp', 'Max Start Time (s)']].rename(columns={'Max Start Time (s)': '_last'})
        if len(st_):
            m = pd.merge_asof(g[['Timestamp']].reset_index(), st_, on='Timestamp', direction='backward',
                              tolerance=pd.Timedelta(days=START_VALID_D)).set_index('index')
            df.loc[g.index, '_tst'] = m['_last'].fillna(0).to_numpy()
    for mid, g in df.groupby('Motor ID'):
        idx = g.index
        for c in SMOOTHED:
            df.loc[idx, c + ' [7d]'] = _rolling(g['Timestamp'], g[c].to_numpy(), LEVEL_WINDOW_D, 'median')
        vmax = np.maximum(g['Vibration DE (mm/s)'].to_numpy(), g['Vibration NDE (mm/s)'].to_numpy())
        bmax = np.maximum(g['Bearing DE Temp (C)'].to_numpy(), g['Bearing NDE Temp (C)'].to_numpy())
        df.loc[idx, 'Vibration Trend (mm/s/30d)'] = _rolling(g['Timestamp'], vmax, TREND_WINDOW_D, 'slope')
        df.loc[idx, 'Bearing Temp Trend (C/30d)'] = _rolling(g['Timestamp'], bmax, TREND_WINDOW_D, 'slope')
    t = None
    if tests is not None and len(tests):
        t = tests.copy(); t['_d'] = pd.to_datetime(t['Date'], errors='coerce'); t['Motor ID'] = t['Motor ID'].astype(str)
        t = t.dropna(subset=['_d']).sort_values('_d')
        for c in ['IR 1min (MOhm)', 'PI', 'MCSA Sideband (dB)']:
            t[c] = pd.to_numeric(t.get(c), errors='coerce')
    # one model call for every reading (same inputs assess() would use)
    fr = np.full(len(df), np.nan)
    if model is not None and len(df):
        X = []
        for _, r in df.iterrows():
            rg = reg[r['Motor ID']]; irr = float(reg_value(rg, 'Rated Current (A)', 100))
            load = (r['Ia'] + r['Ib'] + r['Ic']) / 3 / irr * 100 if irr > 0 else 0
            X.append([load / nema_derate(unbalance(r['Vab'], r['Vbc'], r['Vca'])), unbalance(r['Ia'], r['Ib'], r['Ic']), r['Ambient (C) [7d]']])
        fr = model.predict(np.array(X, dtype=float))
    # daily excess = measured - expected, then its 7-day median per motor
    rise_of = {k: RISE_RATED.get(str(reg_value(v, 'Rise Class', 'B')).upper()[:1], 105.0) for k, v in reg.items()}
    raw_ex = df['Stator Temp (C)'].to_numpy() - (df['Ambient (C)'].to_numpy() + fr * df['Motor ID'].map(rise_of).to_numpy())
    df['_ex'] = raw_ex
    df['_run'] = [((r['Ia'] + r['Ib'] + r['Ic']) / 3 / float(reg_value(reg[r['Motor ID']], 'Rated Current (A)', 100)) * 100) >= 5 for _, r in df.iterrows()]
    df['_ex7'] = np.nan
    for mid, g in df[df['_run']].groupby('Motor ID'):
        df.loc[g.index, '_ex7'] = _rolling(g['Timestamp'], g['_ex'].to_numpy(), LEVEL_WINDOW_D, 'median')
    tgroups = {k: g for k, g in t.groupby('Motor ID')} if t is not None else {}
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        rd = r.to_dict()
        for c in SMOOTHED:
            rd[c] = r[c + ' [7d]']
        rd['Max Start Time (s)'] = r['_tst']
        tv = {}
        if t is not None:
            tg = tgroups.get(r['Motor ID'], t.iloc[0:0])
            tt = tg[(tg['_d'] <= r['Timestamp']) & (tg['_d'] > r['Timestamp'] - pd.Timedelta(days=TEST_VALID_D))]
            for c in ['IR 1min (MOhm)', 'PI', 'MCSA Sideband (dB)']:
                v = tt[c].dropna()
                tv[c] = float(v.iloc[-1]) if len(v) else None
        a = assess(rd, reg[r['Motor ID']], model, tv,
                   {'Vibration Trend (mm/s/30d)': r['Vibration Trend (mm/s/30d)'], 'Bearing Temp Trend (C/30d)': r['Bearing Temp Trend (C/30d)']},
                   exp_frac=None if np.isnan(fr[i]) else float(fr[i]),
                   excess=None if pd.isna(r['_ex7']) else float(r['_ex7']))
        a['Timestamp'] = r['Timestamp']
        a['Tests'] = tv
        rows.append(a)
    return rows


def latest(rows):
    last = {}
    for a in rows:
        if a['Motor ID'] not in last or a['Timestamp'] >= last[a['Motor ID']]['Timestamp']:
            last[a['Motor ID']] = a
    return sorted(last.values(), key=lambda a: -a['Overall Risk (%)'])


def historical_study(rows, events=None, alert_threshold=70.0, lookback_days=365, gap_days=120):
    tl = pd.DataFrame([{'Motor ID': a['Motor ID'], 'Date': pd.Timestamp(a['Timestamp']).floor('D'), 'Overall Risk (%)': a['Overall Risk (%)'],
                        'Status': a['Status'], 'Governing Mode': a['Governing Mode']} for a in rows])
    tl = tl.sort_values(['Motor ID', 'Date']).groupby(['Motor ID', 'Date'], as_index=False).last()
    episodes = []
    for m, g in tl[tl['Overall Risk (%)'] >= alert_threshold].groupby('Motor ID'):
        cur = None
        for d in g['Date']:
            if cur is not None and (d - cur[2]).days <= gap_days:
                cur[2] = d
            else:
                if cur is not None:
                    episodes.append(cur)
                cur = [m, d, d]
        if cur is not None:
            episodes.append(cur)
    ev = pd.DataFrame(columns=['Motor ID', 'Date', 'Event']) if events is None else events.copy()
    ev['_d'] = pd.to_datetime(ev['Date'], errors='coerce'); ev = ev.dropna(subset=['_d']); ev['Motor ID'] = ev['Motor ID'].astype(str)
    out = []
    for _, e in ev.iterrows():
        prior = [p for p in episodes if p[0] == e['Motor ID'] and p[1] < e['_d'] and (e['_d'] - p[2]).days <= lookback_days]
        first = min(p[1] for p in prior) if prior else None
        gov = tl[(tl['Motor ID'] == e['Motor ID']) & (tl['Date'] < e['_d'])]
        out.append({'Motor ID': e['Motor ID'], 'Event Date': e['_d'].date(), 'Event': e.get('Event', ''),
                    'First Alert Before Event': first.date() if first is not None else None,
                    'Lead Time (days)': (e['_d'] - first).days if first is not None else None,
                    'Mode at Alert': gov[gov['Date'] == first]['Governing Mode'].iloc[0] if first is not None else None,
                    'Caught': 'Yes' if first is not None else 'No'})
    fa = sum(1 for m, st, en in episodes if not ((ev['Motor ID'] == m) & (ev['_d'] >= st) & (ev['_d'] <= en + pd.Timedelta(days=lookback_days))).any())
    return tl, pd.DataFrame(out), fa, len(episodes)


# ---------------------------------------------------------------- demo plant
def demo_register():
    rows = [
        ['M-01', 'Cooling water pump A', 1250, 140, 6.6, 'F', 'B', 'Rolling', 'Rigid', 3, 8],
        ['M-02', 'Cooling water pump B', 1250, 140, 6.6, 'F', 'B', 'Rolling', 'Rigid', 3, 8],
        ['M-03', 'Induced draft fan', 2500, 270, 6.6, 'F', 'B', 'Sleeve', 'Flexible', 2, 25],
        ['M-04', 'Air compressor', 900, 98, 6.6, 'F', 'B', 'Rolling', 'Rigid', 4, 6],
        ['M-05', 'Boiler feed pump', 3200, 340, 6.6, 'F', 'B', 'Sleeve', 'Rigid', 2, 12],
        ['M-06', 'Forced draft fan', 1800, 195, 6.6, 'F', 'B', 'Sleeve', 'Flexible', 2, 20],
        ['M-07', 'Fire water pump', 450, 50, 6.6, 'F', 'B', 'Rolling', 'Rigid', 4, 5],
        ['M-08', 'Crusher drive', 1600, 175, 6.6, 'F', 'B', 'Rolling', 'Rigid', 4, 15],
    ]
    return pd.DataFrame(rows, columns=['Motor ID', 'Service', 'Rated kW', 'Rated Current (A)', 'Rated Voltage (kV)', 'Insulation Class',
                                       'Rise Class', 'Bearing Type', 'Foundation', 'Allowed Starts per Day', 'Rated Start Time (s)'])


def demo_data(seed=11, start='2025-01-01', end='2026-08-31'):
    """Synthetic plant history. Scenarios:
       M-01 DE bearing degradation -> bearing replaced after alarm
       M-03 growing fan unbalance (vibration) -> balancing at outage
       M-04 blocked cooler: stator runs hotter than expected -> cooler cleaned
       M-05 broken rotor bars: start time and MCSA sideband worsen -> rotor repair
       M-08 stator insulation: PI falls, hot summer -> stator ground-fault trip
       M-06 recurring supply voltage unbalance (watch, no event)"""
    rng = np.random.default_rng(seed)
    reg = demo_register(); days = pd.date_range(start, end, freq='D')
    D = pd.Timestamp
    ev = {'M-01': D('2026-02-12'), 'M-03': D('2026-04-20'), 'M-04': D('2025-09-15'), 'M-05': D('2026-06-05'), 'M-08': D('2026-07-28')}

    def ramp(day, t0, t1):
        if day < t0:
            return 0.0
        return min(1.0, (day - t0).days / max((t1 - t0).days, 1)) ** 1.5

    recs = []
    for _, m in reg.iterrows():
        mid = m['Motor ID']; ir = m['Rated Current (A)']
        base_load = {'M-07': 0.0}.get(mid, rng.uniform(0.65, 0.85))
        for day in days:
            doy = day.dayofyear
            amb = 33 + 11 * np.sin((doy - 110) / 365 * 2 * np.pi) + rng.normal(0, 1.5)
            load = base_load + rng.normal(0, 0.03)
            if mid == 'M-07':
                load = 0.75 if day.day == 1 else 0.0          # fire pump: test run once a month
            iu, vu = abs(rng.normal(1.2, 0.4)), abs(rng.normal(0.5, 0.2))
            if mid == 'M-06' and (day.month in (6, 7, 8)):
                vu = abs(rng.normal(1.9, 0.15)); iu = vu * 5.5
            frac = thermal_physics(load * 100 / nema_derate(vu), iu, amb)
            ts = amb + frac * 80 + rng.normal(0, 1.5)
            vib = rng.normal(1.6, 0.15) if m['Rated kW'] > 300 else rng.normal(1.0, 0.1)
            vib_n = vib * rng.uniform(0.7, 0.95)
            bde = amb + 22 + 10 * load + rng.normal(0, 1.2); bnde = bde - 3 + rng.normal(0, 1.2)
            starts = rng.poisson(0.3); tst = (m['Rated Start Time (s)'] * rng.uniform(0.92, 1.05)) if starts else 0.0
            if mid == 'M-01' and day < ev['M-01']:
                f = ramp(day, D('2025-07-01'), D('2026-02-05')); bde += 40 * f; vib += 4.5 * f
            if mid == 'M-03' and day < ev['M-03']:
                f = ramp(day, D('2025-07-01'), D('2026-04-10')); vib += 11.0 * f; vib_n += 6 * f
            if mid == 'M-04' and day < ev['M-04']:
                f = ramp(day, D('2025-03-15'), D('2025-09-05')); ts += 30 * f
            if mid == 'M-05' and day < ev['M-05']:
                f = ramp(day, D('2025-09-01'), D('2026-05-25'))
                if starts == 0 and rng.random() < 0.15:
                    starts = 1
                if starts:
                    tst = m['Rated Start Time (s)'] * (1.0 + 0.7 * f) * rng.uniform(0.97, 1.03)
            if mid == 'M-08' and day < ev['M-08']:
                ts += 8 * ramp(day, D('2026-04-01'), D('2026-07-20'))
            if load < 0.05:
                ts, bde, bnde, vib, vib_n = amb + 2, amb + 1, amb + 1, 0.2, 0.2
            I = load * ir
            ia, ib, ic = I * (1 + iu / 100), I * (1 - iu / 200), I * (1 - iu / 200)
            V = 6.6 * rng.uniform(0.98, 1.02)
            vab, vbc, vca = V * (1 + vu / 100), V * (1 - vu / 200), V * (1 - vu / 200)
            recs.append([day + pd.Timedelta(hours=8), mid, round(ia, 1), round(ib, 1), round(ic, 1), round(vab, 3), round(vbc, 3), round(vca, 3),
                         round(amb, 1), round(ts, 1), round(bde, 1), round(bnde, 1), round(max(vib, 0.1), 2), round(max(vib_n, 0.1), 2),
                         int(starts), round(tst, 1)])
    online = pd.DataFrame(recs, columns=['Timestamp', 'Motor ID', 'Ia', 'Ib', 'Ic', 'Vab', 'Vbc', 'Vca', 'Ambient (C)', 'Stator Temp (C)',
                                         'Bearing DE Temp (C)', 'Bearing NDE Temp (C)', 'Vibration DE (mm/s)', 'Vibration NDE (mm/s)',
                                         'Starts', 'Max Start Time (s)'])
    tests = pd.DataFrame([
        ['2025-02-10', m, ir_, pi_, sb] for m, ir_, pi_, sb in
        [('M-01', 2500, 3.4, 58), ('M-02', 2800, 3.6, 59), ('M-03', 1900, 3.1, 57), ('M-04', 2100, 3.2, 58),
         ('M-05', 2600, 3.5, 56), ('M-06', 2300, 3.3, 57), ('M-07', 3000, 3.8, 60), ('M-08', 900, 2.3, 57)]
    ] + [['2025-11-20', 'M-05', 2500, 3.4, 47], ['2026-03-15', 'M-05', 2450, 3.4, 41],
         ['2026-06-12', 'M-05', 2550, 3.5, 58],
         ['2025-11-20', 'M-08', 420, 1.8, 57], ['2026-05-10', 'M-08', 160, 1.4, 56], ['2026-08-10', 'M-08', 3200, 4.1, 58]],
        columns=['Date', 'Motor ID', 'IR 1min (MOhm)', 'PI', 'MCSA Sideband (dB)'])
    events = pd.DataFrame([[k, v.date().isoformat(), t] for k, v, t in [
        ('M-01', ev['M-01'], 'DE bearing replaced after alarm: outer race spalling found'),
        ('M-03', ev['M-03'], 'Planned outage: fan impeller rebalanced, build-up removed'),
        ('M-04', ev['M-04'], 'Cooler found blocked and cleaned'),
        ('M-05', ev['M-05'], 'Rotor removed: three broken bars repaired'),
        ('M-08', ev['M-08'], 'Stator ground-fault trip: winding rewind')]], columns=['Motor ID', 'Date', 'Event'])
    return {'register': reg, 'online': online, 'tests': tests, 'events': events}
