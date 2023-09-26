import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)


#Streamlit app
st.title('Iris Dataset Classification')
st.write('Using Logistic Regression')
st.write(f'Accuracy: {accuracy * 100:.2f}%')

#User inputs
st.subheader('Predict a new Sample')
sepal_length = st.number_input('Sepal Length (cm):')
sepal_width = st.number_input('Sepal Width (cm):')
petal_length = st.number_input('Petal Length (cm):')
petal_width = st.number_input('Petal Width (cm):')

if st.button('Predict'):
    user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(user_input)
    species = iris.target_names[prediction][0]
    st.write(f'The Predicted species is : {species}')