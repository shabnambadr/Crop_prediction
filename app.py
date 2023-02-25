import streamlit as st

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, render_template, request
import pickle

# Load the sales data
sales_df = pd.read_csv('sales_data.csv')

# Create the feature matrix and target vector
X = sales_df[['item_name', 'day']]
y = sales_df['sales']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, columns=['day', 'item_name'], prefix=['day', 'item_name'])

# Create the linear regression model
model = LinearRegression()

# Train the model on the sales data
model.fit(X, y)

# Calculate the mean squared error and R^2 score for the model
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print('Mean squared error: ', mse)
print('R^2 score:', r2)

# Get input from user
item_name = st.text_input("Enter item name: ")
day = st.text_input('Enter day : ')

# Create new input
X_new = pd.DataFrame({'item_name': [item_name], 'day': [day]})
X_new = pd.get_dummies(X_new, columns=['day', 'item_name'], prefix=['day', 'item_name'])

# Add missing dummy variables
missing_cols = set(X.columns) - set(X_new.columns)
for col in missing_cols:
    X_new[col] = 0

# Ensure columns are in the same order
X_new = X_new[X.columns]

# Predict the sales for the new input
y_new = model.predict(X_new)

print('Predicted sales: ', y_new[0])


# Load trained model from file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set up Flask application
app = Flask(__name__)

# Define route for home page
@app.route("/")
def home():
    return render_template('index.html')

# Define route for prediction page
@app.route("/predict", methods=['POST'])
def predict():
    # Get input from web form
    day = request.form['day']
    item_id = request.form['item_id']

    # Use model to make prediction
    X_new = [[day, item_id]]
    y_new = model.predict(X_new)

    # Render prediction page with predicted sales
    return render_template('predict.html', sales=y_new[0])

# Run Flask application
if __name__ == '_main_':
    app.run(debug=True,port=5001)
