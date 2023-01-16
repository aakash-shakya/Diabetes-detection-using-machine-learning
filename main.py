import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

import streamlit as st

model = joblib.load('diabetes-prediction-model.pkl')


def user_data(raw_input_data):
    scaler = StandardScaler()
    input_data = np.asarray(raw_input_data)
    reshaped_input_data = input_data.reshape(1,-1)

    std_input_data = scaler.fit_transform(reshaped_input_data)

    real_prediction = model.predict(std_input_data)

    if real_prediction == 1:
        return('person is diabetic')
    else:
        return('person is not diabetic')


# giving a title
st.title('Diabetes Prediction Web App')


# getting the input data from the user


Pregnancies = st.text_input('Number of Pregnancies', placeholder="Enter the number of pregnancies")
Glucose = st.text_input('Glucose Level', placeholder="Enter the Glucose Level")
BloodPressure = st.text_input('Blood Pressure value', placeholder="Enter the Blood Pressure value")
SkinThickness = st.text_input('Skin Thickness value', placeholder="Enter the Skin Thickness value")
Insulin = st.text_input('Insulin Level',    placeholder="Enter the Insulin Level")
BMI = st.text_input('BMI value', placeholder="Enter the BMI value")
DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', placeholder="Enter the Diabetes Pedigree Function value")
Age = st.text_input('Age of the Person', placeholder="Enter the Age of the Person")


# code for Prediction
diagnosis = ''

# creating a button for Prediction

if st.button('Diabetes Test Result'):
    diagnosis = user_data([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    
st.success(diagnosis)