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
rf_model = RandomForestRegressor()
rf_model = joblib.load('rf_model.pkl')
lr_model = LinearRegression()
lr_model = joblib.load('lr_model.pkl')

# Get user input
rainfall = st.sidebar.slider('Rainfall', 0, 500, 100)
ph = st.sidebar.slider('pH', 3.5, 9.0, 7.0)
temperature = st.sidebar.slider('Temperature', 0, 50, 25)
n_content = st.sidebar.slider('N Content', 0, 200, 100)
p_content = st.sidebar.slider('P Content', 0, 200, 100)
k_content = st.sidebar.slider('K Content', 0, 200, 100)

# Make predictions
rf_prediction = rf_model.predict([[rainfall, ph, temperature, n_content, p_content, k_content]])
lr_prediction = lr_model.predict([[rainfall, ph, temperature, n_content, p_content, k_content]])

# Calculate accuracy
rf_accuracy = r2_score(y_true, rf_prediction)
lr_accuracy = r2_score(y_true, lr_prediction)

# Plot bar chart
fig, ax = plt.subplots()
model_names = ['Random Forest', 'Linear Regression']
accuracy_values = [rf_accuracy, lr_accuracy]
ax.bar(model_names, accuracy_values)
ax.set_title('Accuracy Comparison')
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
st.pyplot(fig)

