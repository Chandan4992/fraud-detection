# 💳 Credit Card Fraud Detection Web App

> 🔍 A production-ready ML web app for detecting fraudulent transactions in real-time using LightGBM and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff69b4.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-card-fraud-detection-mlp.streamlit.app/)

---

## 🚀 Live Demo
👉 https://credit-card-fraud-detection-mlp.streamlit.app/

---

## 📸 App Screenshots

### 🔹 Main Interface
![App UI](screenshots/ui.png)

### 🔹 Prediction Result
![Prediction](screenshots/result.png)

---

## 📌 Project Overview
This project predicts whether a transaction is fraudulent using a Machine Learning model.  
It uses a trained **LightGBM model** and provides a simple UI using **Streamlit** for real-time predictions.

---

## ✨ Features
- Real-time fraud prediction  
- Interactive user interface  
- Fraud probability score  
- Feature engineering (distance + time)  
- Transaction map visualization  

---

## 💻 Technology Stack
- Python  
- LightGBM  
- Streamlit  
- Pandas, NumPy  
- Scikit-learn  

---

## ⚙️ Methodology

### 🔹 Data Processing
- Cleaned dataset  
- Removed unnecessary columns  

### 🔹 Feature Engineering
- Distance calculation (Haversine)  
- Time features (hour, day, month, weekday)  

### 🔹 Model Training
- Used LightGBM for fast and accurate predictions  
- Handled imbalance using weighting  

### 🔹 Model Performance
The model achieved a very high AUC score on validation data.  
Further validation on real-world data is recommended.

---

## 🧠 Key Learnings
- Handling imbalanced datasets  
- Feature engineering using time & location  
- Building ML web apps  
- Deploying models online  

---

## ⚠️ Limitations
- Model may not generalize perfectly to real-world data  
- Dataset imbalance can bias predictions  
- Needs retraining with updated data  

---

## 🚀 Future Improvements
- Add SHAP explainability  
- Improve model accuracy  
- Add analytics dashboard  
- Integrate real-time APIs  

---

## 🚀 Setup and Installation

```bash
git clone https://github.com/Ashish-kharde1/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
streamlit run app.py