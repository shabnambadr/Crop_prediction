import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv', dtype={'yield': np.float64})
data['yield'] = pd.to_numeric(data['yield'], errors='coerce')

# Drop any rows with NaN values
data.dropna(inplace=True)

# Separate the input features and target variable
X = data.drop('yield', axis=1)
y = data['yield']

# Train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X, y)

# Train the random forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

# Define the slider inputs
N = st.slider('Nitrogen content', 0.0, 250.0, 100.0, 1.0)
P = st.slider('Phosphorous content', 0.0, 200.0, 50.0, 1.0)
K = st.slider('Potassium content', 0.0, 250.0, 100.0, 1.0)
temperature = st.slider('Temperature', 0.0, 50.0, 25.0, 1.0)
humidity = st.slider('Humidity', 0.0, 100.0, 50.0, 1.0)
pH = st.slider('pH', 0.0, 14.0, 7.0, 0.1)
rainfall = st.slider('Rainfall', 0.0, 300.0, 100.0, 1.0)

# Make predictions using both models
lr_pred = lr_model.predict([[N, P, K, temperature, humidity, pH, rainfall]])
rf_pred = rf_model.predict([[N, P, K, temperature, humidity, pH, rainfall]])

# Print the predicted crop based on the more accurate model
if lr_pred < rf_pred:
    predicted_model = 'Random Forest'
    predicted_crop = rf_pred[0]
else:
    predicted_model = 'Linear Regression'
    predicted_crop = lr_pred[0]

st.write('Predicted crop based on', predicted_model, 'model:', predicted_crop)

# Evaluate the accuracy of both models using RMSE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

# Plot the accuracy levels of the two models
fig, ax = plt.subplots()
ax.bar(['Linear Regression', 'Random Forest'], [lr_rmse, rf_rmse])
ax.set_xlabel('Model')
ax.set_ylabel('RMSE')
ax.set_title('Accuracy Comparison')
st.pyplot(fig)







