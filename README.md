# MotorGuard AI

HV induction motor condition and risk monitor. Part of the PowerGuard AI platform. Confidential and proprietary.

**Streamlit main file:** `MotorGuard_AI.app.py`
**Secrets:** `APP_USERNAME`, `APP_PASSWORD`

## Pages
- **Plant Overview:** results collected from the analysis pages, or the demo plant (8 motors).
- **Motor Risk & FMEA:** latest reading of one motor → risk for 7 failure modes, overall risk, FMEA and RPN.
- **Stator & Cooling / Bearings & Vibration / Supply & Starting:** quick checks of one subsystem.
- **Combined Report** and **Historical Study** (demo or client CSV files).

## Files
- `motorguard_core.py`: engine (IEC 60034-1 / NEMA MG1 thermal limits, IEEE 43, ISO 20816-3, NEMA MG1 unbalance derating, MCSA).
- `thermal_model.pkl`: expected stator temperature model (`train_thermal.py`). Retrained automatically if it cannot be loaded.
