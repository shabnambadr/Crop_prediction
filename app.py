import streamlit as st
import joblib
model=joblib.load('IRIS CLASSIFIER')
st.title('IRIS CLASSIFIER')
ip1=st.slider('Enter Sepal Length:', min_value=0, max_value=10, step=0.1, format=%f)
ip2=st.slider('Enter Sepal Width:', min_value=0, max_value=10, step=0.1, format=%f)
ip3=st.slider('Enter Petal Length:', min_value=0, max_value=10, step=0.1, format=%f)
ip4=st.slider('Enter Petal Width:', min_value=0, max_value=10, step=0.1, format=%f)
op=model.predict([ip1,ip2,ip3,ip4])
if st.button('Predict'):
  st.title(op[0])
