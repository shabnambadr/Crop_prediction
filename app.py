import sklearn
import streamlit as st
import joblib
model=joblib.load('IRIS CLASSIFIER new')
st.title('IRIS CLASSIFIER')
st.subheader('Sepal Length')
ip1=st.slider('Enter a value between 0 & 10:', min_value=0, max_value=10, step=0.1)
st.subheader('Sepal Width')
ip2=st.slider('Enter a value between 0 & 10:', min_value=0, max_value=10, step=0.1)
st.subheader('Petal Length')
ip3=st.slider('Enter a value between 0 & 10:', min_value=0, max_value=10, step=0.1)
st.subheader('Petal Length')
ip4=st.slider('Enter a value between 0 & 10:', min_value=0, max_value=10, step=0.1)
op=model.predict([ip1,ip2,ip3,ip4])
if st.button('Predict'):
  st.title(op[0])

  
