import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# --- PAGE SETUP ---
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('lgb_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# --- HAVERSINE FUNCTION ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# --- LOAD MAPPINGS ---
@st.cache_data
def get_mappings():
    df = pd.read_csv('fraudTrain.csv')
    mappings = {}
    cols = ['category','gender','city','state','job']
    for col in cols:
        vals = sorted(df[col].unique())
        mappings[col] = {v:i for i,v in enumerate(vals)}
        mappings[col+"_list"] = vals
    return mappings

mappings = get_mappings()

# --- UI ---
st.title("💳 Credit Card Fraud Detection System")

st.sidebar.header("Input Details")

category = st.sidebar.selectbox("Category", mappings['category_list'])
amt = st.sidebar.number_input("Amount", value=100.0)

date = st.sidebar.date_input("Date", datetime.now())
time = st.sidebar.time_input("Time", datetime.now().time())

gender = st.sidebar.selectbox("Gender", mappings['gender_list'])
age = st.sidebar.slider("Age", 18, 120, 30)
job = st.sidebar.selectbox("Job", mappings['job_list'])

city = st.sidebar.selectbox("City", mappings['city_list'])
state = st.sidebar.selectbox("State", mappings['state_list'])
city_pop = st.sidebar.number_input("City Population", value=50000)

lat = st.sidebar.number_input("Cardholder Lat", value=34.05)
lon = st.sidebar.number_input("Cardholder Lon", value=-118.24)
mlat = st.sidebar.number_input("Merchant Lat", value=34.15)
mlon = st.sidebar.number_input("Merchant Lon", value=-118.44)

dt = datetime.combine(date, time)

# --- PREDICTION ---
if st.button("Detect Fraud"):

    # Distance
    distance = haversine(lat, lon, mlat, mlon)

    # Features
    features = pd.DataFrame([{
        'category': mappings['category'][category],
        'amt': amt,
        'gender': mappings['gender'][gender],
        'city': mappings['city'][city],
        'state': mappings['state'][state],
        'lat': lat,
        'long': lon,
        'city_pop': city_pop,
        'job': mappings['job'][job],
        'unix_time': int(dt.timestamp()),
        'merch_lat': mlat,
        'merch_long': mlon,
        'age': age,
        'distance': distance,
        'hour': dt.hour,
        'day': dt.day,
        'month': dt.month,
        'weekday': dt.weekday()
    }])

    # ✅ IMPORTANT: same transform as training
    features['amt'] = np.log1p(features['amt'])

    # Model prediction (LightGBM Booster)
    prob = model.predict(features)[0]

    # 🚨 Smart rules (real-world logic)
    if amt > 10000:
        prob = max(prob, 0.7)

    if distance > 100:
        prob = max(prob, 0.6)

    # Final decision
    is_fraud = prob > 0.5

    # --- OUTPUT ---
    st.subheader("Result")

    if is_fraud:
        st.error("🚨 High Risk Fraud Detected")
    else:
        st.success("✅ Low Risk Transaction")

    st.metric("Fraud Probability", f"{prob:.6f}")
    st.progress(float(prob))

    # Map
    st.subheader("Transaction Map")
    map_df = pd.DataFrame({
        "lat": [lat, mlat],
        "lon": [lon, mlon]
    })
    st.map(map_df)

    # Debug info
    with st.expander("Details"):
        st.write("Distance (km):", distance)
        st.write("Final Probability:", prob)
        st.dataframe(features)

else:
    st.info("Fill details and click Detect Fraud")