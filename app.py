import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle

# Load the trained models and scalers
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('scaler_hdi.pkl', 'rb') as f:
    scaler_hdi = pickle.load(f)

with open('le.pkl', 'rb') as f:
    le = pickle.load(f)

with open('rf.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)

# Streamlit app
st.title("HDI Prediction App")

st.write("Enter the values for the following features to predict the HDI class and score:")

# Create input fields for the features
life_expectancy = st.number_input('Life expectancy', min_value=0.0, max_value=100.0, value=0.0)
expected_schooling = st.number_input('Expected years of Schooling', min_value=0.0, max_value=25.0, value=12.0)
mean_schooling = st.number_input('Mean years of Schooling', min_value=0.0, max_value=20.0, value=8.0)
gni = st.number_input('GNI($)', min_value=0, value=10000)

# Create a button to make predictions
if st.button('Predict'):
    # Create a DataFrame from the user input
    input_features = {
        'Life expectancy': life_expectancy,
        'Expected years of Schooling': expected_schooling,
        'Mean years of Schooling': mean_schooling,
        'GNI($)': gni
    }
    input_df = pd.DataFrame([input_features])

    # Scale the input features
    input_scaled_classification = scaler.transform(input_df)
    input_scaled_regression = scaler_hdi.transform(input_df)

    # Predict the HDI class
    prediction_encoded = rf.predict(input_scaled_classification)
    prediction_class = le.inverse_transform(prediction_encoded)

    # Predict the HDI score
    predicted_hdi_score = regressor.predict(input_scaled_regression)

    st.write(f"Predicted HDI Class: {prediction_class[0]}")
    st.write(f"Predicted HDI Score: {predicted_hdi_score[0]:.4f}")
