# Importing required libraries                                                                             
import pandas as pd                                                                                        
from sklearn.linear_model import LinearRegression                                                          
from sklearn.preprocessing import OneHotEncoder                                                            
import pickle                                                                                              
                                                                                                           
# Load dataset from CSV file                                                                               
data = pd.read_csv("data/Crop_recommendation.csv")                                                         
                                                                                                           
# One-hot encoding the crop names                                                                          
enc = OneHotEncoder(sparse=False)                                                                          
encoded_y = enc.fit_transform(data[['yield']])                                                             
encoded_y = pd.DataFrame(encoded_y, columns=enc.get_feature_names_out(['yield']))                          
data = pd.concat([data, encoded_y], axis=1)                                                                
                                                                                                           
# Creating the input and output variables for the regression model                                         
X = data[['humidity', 'temperature', 'rainfall']]                                                          
y = data.drop(['humidity', 'temperature', 'rainfall', 'yield'], axis=1)                                    
                                                                                                           
# Creating the regression model and fitting it to the input and output variables                           
model = LinearRegression()                                                                                 
model.fit(X, y)                                                                                            
                                                                                                           
# Saving the model to a file                                                                               
with open('crop_yield_prediction_model.pkl', 'wb') as f:                                                   
    pickle.dump(model, f)                                                                                  
                                                                                                           
# Get input from user for new test data                                                                    
new_data = pd.DataFrame(columns=['humidity', 'temperature', 'rainfall'])                                   
new_data.loc[0] = [float(input("Enter humidity value: ")),                                                 
                   float(input("Enter temperature value: ")),                                              
                   float(input("Enter rainfall value: "))]                                                 
                                                                                                           
# Predicting the yield for the input parameters using the trained model                                    
predicted_yield = model.predict(new_data)                                                                  
                                                                                                           
# Dropping the additional column from the encoded data                                                     
encoded_y = data.drop(['humidity', 'temperature', 'rainfall', 'yield'], axis=1)                            
                                                                                                           
# Inverse transforming the predicted yield to get the original crop name                                   
inverse_predicted_yield = enc.inverse_transform(predicted_yield[:, :22])                                   
                                                                                                           
# Printing the predicted yield                                                                             
print('The crop with the maximum yield based on the given parameters is:', inverse_predicted_yield[0][0])  



