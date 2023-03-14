import pickle

# Load the trained model from the .py file
model = 'lr_model.py'

# Define the file name for the .pkl file
file_name = 'lr_model.pkl'

# Save the trained model as a .pkl file
with open(file_name, 'wb') as file:
    pickle.dump(model, file)
