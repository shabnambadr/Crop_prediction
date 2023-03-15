import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv('Crop_recommendation.csv')

# Load trained models
rf_model = joblib.load('rf_model.pkl')
lr_model = joblib.load('lr_model.pkl')

# Get user input
humidity = 50.0
temperature = 25.0
rainfall = 100.0
ph = 7.0
N = 50.0
P = 25.0
K = 75.0

# Make predictions
rf_prediction = rf_model.predict([[humidity, temperature, rainfall, ph, N, P, K]])
lr_prediction = lr_model.predict([[humidity, temperature, rainfall, ph, N, P, K]])

# Calculate accuracy
y_true = data['label']
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
plt.show()




