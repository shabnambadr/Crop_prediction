import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lr_model import LinearRegression
from rf_model import RandomForestRegressor

# Load your dataset or input data
data = pd.read_csv('Crop_recommendation.csv')
X_train, X_test, y_train, y_test = train_test_split(data.drop('yield'), data['yield'], test_size=0.3, random_state=42)

# Create instances of the models and fit them with your dataset
lr_model = LinearRegression()
rf_model = RandomForestRegressor()
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predict using the models
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Compute the accuracy of each model
lr_acc = np.mean((lr_preds - y_test)**2)
rf_acc = np.mean((rf_preds - y_test)**2)

# Plot the predicted values vs. the actual values
fig, ax = plt.subplots()
ax.scatter(y_test, lr_preds, color='red')
ax.scatter(y_test, rf_preds, color='blue')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.legend(['Linear Regression', 'Random Forest'])

# Display the plot in Streamlit
st.pyplot(fig)

# Display the accuracy of each model
st.write("Linear Regression Accuracy:", lr_acc)
st.write("Random Forest Accuracy:", rf_acc)





