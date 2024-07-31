import joblib

scaler = joblib.load('scaler100.pkl')
print(scaler.mean_, scaler.scale_)