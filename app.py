import streamlit as st
import joblib
import pandas as pd
from sklearn import datasets
st.title('IRIS CLASSIFIER')
st.subheader('Sepal Length')
ip1=st.slider('Enter a value between 0 & 10:', min_value=0.0, max_value=10.0, step=0.1, key=1)
st.subheader('Sepal Width')
ip2=st.slider('Enter a value between 0 & 10:', min_value=0.0, max_value=10.0, step=0.1, key=2)
st.subheader('Petal Length')
ip3=st.slider('Enter a value between 0 & 10:', min_value=0.0, max_value=10.0, step=0.1, key=3)
st.subheader('Petal Length')
ip4=st.slider('Enter a value between 0 & 10:', min_value=0.0, max_value=10.0, step=0.1, key=4)
iris=datasets.load_iris()
x=iris.data
y=iris.target
df=pd.DataFrame(iris.data)
df['Target']=iris.target
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x,y)
op=model.predict([[ip1,ip2,ip3,ip4]])
if op[0]==0:
  out="Iris setosa"
elif op[0]==1:
  out="Iris versicolor"
elif op[0]==2:
  out="Iris virginica"
if st.button('Predict'):
  st.title(out)
