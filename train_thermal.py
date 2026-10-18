"""Train the expected-stator-temperature model (RandomForest regression on physics-based data)."""
import pickle
from sklearn.ensemble import RandomForestRegressor
from motorguard_core import make_thermal_training

X, y = make_thermal_training()
m = RandomForestRegressor(n_estimators=60, max_depth=11, min_samples_leaf=8, random_state=0, n_jobs=-1).fit(X[:16000], y[:16000])
print('R2 test:', round(m.score(X[16000:], y[16000:]), 4))
pickle.dump(m, open('thermal_model.pkl', 'wb'))
