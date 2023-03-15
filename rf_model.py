# Importing required libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import pickle
import streamlit as st

# Load dataset from CSV file
data = pd.read_csv("Crop_recommendation.csv")

# One-hot encoding the crop names
enc = OneHotEncoder(sparse=False)
encoded_y = enc.fit_transform(data[['yield']])
encoded_y = pd.DataFrame(encoded_y, columns=enc.get_feature_names_out(['yield']))
data = pd.concat([data, encoded_y], axis=1)

# Creating the input and output variables for the regression model
X = data[['humidity', 'temperature', 'rainfall', 'ph', 'N', 'P', 'K']]
y = data.drop(['humidity', 'temperature', 'rainfall', 'ph', 'N', 'P', 'K', 'yield'], axis=1)

# Creating the regression model and fitting it to the input and output variables
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

# Saving the model to a file
with open('crop_yield_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Add input widgets for new test data
st.sidebar.header('Input Parameters')
humidity = st.sidebar.slider('Humidity', 0.0, 100.0, 50.0, 1.0, key= humidity_slider)
temperature = st.sidebar.slider('Temperature', 0.0, 50.0, 25.0, 1.0, key= temperature_slider)
rainfall = st.sidebar.slider('Rainfall', 0.0, 300.0, 100.0, 1.0, key= rainfall_slider)
ph = st.sidebar.slider('pH', 3.0, 10.0, 7.0, 0.1, key= ph_slider)
N = st.sidebar.slider('Nitrogen Content', 0, 250, 100, 1, key= N_slider)
P = st.sidebar.slider('Phosphorous Content', 0, 250, 100, 1, key= P_slider)
K = st.sidebar.slider('Potassium Content', 0, 250, 100, 1, key= K_slider)

# Create a button to trigger prediction
if st.sidebar.button('Predict Yield'):
    # Predicting the yield for the input parameters using the trained model
    new_data = pd.DataFrame({'humidity': [humidity], 'temperature': [temperature], 'rainfall': [rainfall], 'ph': [ph], 'N': [N], 'P': [P], 'K': [K]})
    predicted_yield = model.predict(new_data)

    # Dropping the additional column from the encoded data
    encoded_y = data.drop(['humidity', 'temperature', 'rainfall', 'ph', 'N', 'P', 'K', 'yield'], axis=1)

    # Inverse transforming the predicted yield to get the original crop name
    inverse_predicted_yield = enc.inverse_transform(predicted_yield[:, :22])

    # Displaying the predicted yield
    st.header('Predicted Yield')
    st.write('The crop with the maximum yield based on the given parameters is:', inverse_predicted_yield[0][0])

 



