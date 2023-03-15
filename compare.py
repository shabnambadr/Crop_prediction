import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load data
data = pd.read_csv('Crop_recommendation.csv')

# Load trained models
rf_model = joblib.load('rf_model.pkl')
lr_model = joblib.load('lr_model.pkl')

# Get user input
humidity = st.sidebar.slider('Humidity', 0.0, 100.0, 50.0, 1.0)
temperature = st.sidebar.slider('Temperature', 0.0, 50.0, 25.0, 1.0)
rainfall = st.sidebar.slider('Rainfall', 0.0, 300.0, 100.0, 1.0)
ph = st.sidebar.slider('pH', 3.0, 10.0, 7.0, 0.1)
N = st.sidebar.slider('Nitrogen', 0.0, 150.0, 50.0, 1.0)
P = st.sidebar.slider('Phosphorous', 0.0, 100.0, 25.0, 1.0)
K = st.sidebar.slider('Potassium', 0.0, 200.0, 75.0, 1.0)

# Make predictions
rf_prediction = rf_model.predict([[humidity, temperature, rainfall, ph, N, P, K]])
lr_prediction = lr_model.predict([[humidity, temperature, rainfall, ph, N, P, K]])

# Display predicted values to user
st.write('Random Forest predicted yield:', rf_prediction[0])
st.write('Linear Regression predicted yield:', lr_prediction[0])



