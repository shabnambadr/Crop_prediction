from flask import Flask, render_template, request
import pickle

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
