import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection System")

# --- CHECK FILES ---
if not os.path.exists("model_bundle.pkl"):
    st.error("❌ model_bundle.pkl not found")
    st.stop()

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('model_bundle.pkl', 'rb') as f:
        data = pickle.load(f)
    return data["model"], data["preprocessor"]

model, preprocessor = load_model()

# --- LOAD OPTIONS ---
@st.cache_data
def load_options():
    df = pd.read_csv('fraudTrain.csv', nrows=5000)
    return {
        "category": sorted(df["category"].unique()),
        "gender": sorted(df["gender"].unique()),
        "city": sorted(df["city"].unique()),
        "state": sorted(df["state"].unique()),
        "job": sorted(df["job"].unique())
    }

options = load_options()

# --- DISTANCE ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# --- UI ---
st.sidebar.header("Input Details")

category = st.sidebar.selectbox("Category", options['category'])
amt = st.sidebar.number_input("Amount", value=100.0)

date = st.sidebar.date_input("Date", datetime.now())
time = st.sidebar.time_input("Time", datetime.now().time())

gender = st.sidebar.selectbox("Gender", options['gender'])
age = st.sidebar.slider("Age", 18, 120, 30)
job = st.sidebar.selectbox("Job", options['job'])

city = st.sidebar.selectbox("City", options['city'])
state = st.sidebar.selectbox("State", options['state'])
city_pop = st.sidebar.number_input("City Population", value=50000)

lat = st.sidebar.number_input("Cardholder Lat", value=34.05)
lon = st.sidebar.number_input("Cardholder Lon", value=-118.24)
mlat = st.sidebar.number_input("Merchant Lat", value=34.15)
mlon = st.sidebar.number_input("Merchant Lon", value=-118.44)

dt = datetime.combine(date, time)

# --- PREDICT ---
if st.button("Detect Fraud"):

    distance = haversine(lat, lon, mlat, mlon)

    features = pd.DataFrame([{
        'category': category,
        'amt': amt,
        'gender': gender,
        'city': city,
        'state': state,
        'lat': lat,
        'long': lon,
        'city_pop': city_pop,
        'job': job,
        'merch_lat': mlat,
        'merch_long': mlon,
        'age': age,
        'distance': distance,
        'hour': dt.hour,
        'day': dt.day,
        'month': dt.month,
        'weekday': dt.weekday()
    }])

    features_processed = preprocessor.transform(features)
    prob = model.predict(features_processed)[0]

    is_fraud = prob > 0.5

    st.subheader("Result")

    if is_fraud:
        st.error("🚨 Fraud Detected")
    else:
        st.success("✅ Safe Transaction")

    st.metric("Fraud Probability", f"{prob:.4f}")
    st.progress(float(prob))

    st.map(pd.DataFrame({
        "lat": [lat, mlat],
        "lon": [lon, mlon]
    }))

else:
    st.info("Enter details and click Detect Fraud")