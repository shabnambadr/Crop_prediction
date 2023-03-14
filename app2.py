import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Create and train the random forest model
dt = RandomForestRegressor()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# Display the results in a pie chart
data = {'Model': ['Logistic Regression', 'Random Forest'
df = pd.DataFrame(data)
fig = px.pie(df, values='Accuracy', names='Model')
st.plotly_chart(fig)

# Display the higher accuracy model
if lr_acc > dt_acc:
    st.write('Logistic Regression has higher accuracy: ', lr_acc)
else:
    st.write('Random Forest has higher accuracy: ', dt_acc)

