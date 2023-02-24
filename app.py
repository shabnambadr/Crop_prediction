import pandas as pd
import joblib

# Load data from CSV file
data = pd.read_csv('sales_data.csv')

# Preprocess data


X = data[['day', 'item_id']] # Input variables
y = data['sales'] # Target variable
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

# Define linear regression model
model = LinearRegression()
# Train model on training data
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score


# Evaluate model on testing data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error: ', mse)
print('R-squared: ', r2)
# Predict sales for new data
item_id = int(input("Enter item id: "))
day = int(input("Enter day (1-7): "))

X_new = [[day, item_id]]
y_new = model.predict(X_new)

print('Predicted sales: ', y_new)
joblib.dump(model, 'model.pkl')
from flask import Flask, render_template, request
import pickle
import streamlit as st
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
