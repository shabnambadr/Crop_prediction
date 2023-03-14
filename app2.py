import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('Crop_recommendation.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data[['ph', 'humidity', 'rainfall', 'temperature', 'N', 'P', 'K']],
    data['crop'],
    test_size=0.3,
    random_state=42
)

# Linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Random forest model
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

# Prediction and accuracy evaluation
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred.round())
rf_accuracy = accuracy_score(y_test, rf_pred.round())

# Plot pie chart comparing accuracies
fig, ax = plt.subplots(figsize=(5,5))
accuracy = [lr_accuracy, rf_accuracy]
models = ['Linear Regression', 'Random Forest']
ax.pie(accuracy, labels=models, autopct='%1.2f%%')
ax.set_title('Model Accuracy')
st.pyplot(fig)
