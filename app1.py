import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle

# Load models and scalers as before ...
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

st.set_page_config(page_title="HDI Prediction", page_icon="🌍", layout="centered")
st.title("🌍 HDI Prediction")
st.markdown("""
Enter the values for the features below to predict the **HDI class** and ***score***.
""")

# Layout: Inputs in Sidebar
st.sidebar.header("Input Features")
life_expectancy = st.sidebar.slider(
    'Life expectancy', 0.0, 100.0, 70.0, help="Average number of years a person is expected to live")
expected_schooling = st.sidebar.slider(
    'Expected years of Schooling', 0.0, 25.0, 12.0, help="Number of years a child of school entrance age is expected to study(Country Average)")
mean_schooling = st.sidebar.slider(
    'Mean years of Schooling', 0.0, 20.0, 8.0, help="Average number of completed years of education of a country's population")
gni = st.sidebar.number_input(
    'GNI ($)', 0, 100000, 10000, step=500, help="Gross National Income per capita in USD (2017 PPP)")

if st.sidebar.button('Predict'):
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

    # Scaling and predictions as before...

    st.subheader("Prediction Result")
    st.success(f"**Predicted HDI Class:** {prediction_class[0]}")
    st.info(f"**Predicted HDI Score:** {predicted_hdi_score[0]:.4f}")
else:
    st.write("Set feature values in the sidebar and click 'Predict'.")

st.markdown("---")
st.caption("HDI: Human Development Index | Model predictions are for educational/demo use.")

