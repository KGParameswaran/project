import streamlit as st
import pandas as pd
import pickle

# =======================
# Load Models & Encoders
# =======================
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

# =======================
# Page Config
# =======================
st.set_page_config(page_title="HDI Prediction", page_icon="🌍", layout="wide")

# =======================
# Custom Background Style
# =======================
page_bg = """
<style>
.stApp {
    background: linear-gradient(to right, #83a4d4, #b6fbff);
    background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =======================
# Sidebar Styling
# =======================
sidebar_bg = """
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #83a4d4, #b6fbff);
}
[data-testid="stSidebar"] * {
    color: black;
}
</style>
"""
st.markdown(sidebar_bg, unsafe_allow_html=True)

# =======================
# Top Navbar Styling
# =======================
topbar_bg = """
<style>
header[data-testid="stAppHeader"] {
    background: linear-gradient(to right, #4facfe, #00f2fe);
}
</style>
"""
st.markdown(topbar_bg, unsafe_allow_html=True)


# =======================
# Custom Widget Styling
# =======================
custom_widgets = """
<style>
/* Style Gross National Income input box */
div[data-baseweb="input"] > input[aria-label="Gross National Income per capita (USD)"] {
    background-color: white !important;
    color: black !important;
    border-radius: 8px !important;
    border: 1px solid #ccc !important;
    padding: 6px !important;
}

/* Style Predict HDI button */
div.stButton > button {
    background-color: white !important;
    color: black !important;
    border-radius: 10px !important;
    border: 1px solid #4facfe !important;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #f0f0f0 !important;
    border: 1px solid #00f2fe !important;
}
</style>
"""
st.markdown(custom_widgets, unsafe_allow_html=True)


# =======================
# GNI Input Styling
# =======================
gni_style = """
<style>
div[data-baseweb="input"] input[aria-label="Gross National Income per capita (USD)"] {
    background-color: white !important;
    color: black !important;
    border-radius: 6px !important;
    border: 1px solid #ccc !important;
    padding: 6px !important;
}
</style>
"""
st.markdown(gni_style, unsafe_allow_html=True)


# =======================
# Main Title & Description
# =======================
st.title("🌍 Human Development Index (HDI) Prediction")
st.markdown("""
This tool predicts a country’s **HDI Class** (Low, Medium, High, Very High)  
and its **HDI Score** based on key socioeconomic indicators.  

👉 Use the sidebar to input values and click **Predict**.
""")

# =======================
# What is HDI?
# =======================
with st.expander("ℹ️ What is the Human Development Index (HDI)?", expanded=True):
    st.write("""
    The **Human Development Index (HDI)** is a summary measure of human development
    published by the **United Nations Development Programme (UNDP)**.  

    It combines three key dimensions:
    - 📈 **Life Expectancy**: A long and healthy life  
    - 🎓 **Education**: Expected and mean years of schooling  
    - 💰 **Income**: Gross National Income (GNI) per capita  

    HDI values range from **0 to 1**, and countries are grouped into:
    - **Low Human Development** (below 0.55)  
    - **Medium Human Development** (0.55–0.70)  
    - **High Human Development** (0.70–0.80)  
    - **Very High Human Development** (above 0.80)  
    """)


# =======================
# Sidebar Inputs
# =======================
st.sidebar.header("🔧 Input Features")

life_expectancy = st.sidebar.slider(
    "Life Expectancy (years)", 
    min_value=0.0, max_value=100.0, value=70.0, step=0.1,
    help="Average number of years a person is expected to live"
)

expected_schooling = st.sidebar.slider(
    "Expected Years of Schooling", 
    min_value=0.0, max_value=25.0, value=12.0, step=0.1,
    help="Years a child entering school is expected to study"
)

mean_schooling = st.sidebar.slider(
    "Mean Years of Schooling", 
    min_value=0.0, max_value=20.0, value=8.0, step=0.1,
    help="Average completed years of education in the population"
)

gni = st.sidebar.number_input(
    "Gross National Income per capita (USD)", 
    min_value=0, max_value=100000, value=10000, step=500,
    help="GNI per capita in 2017 PPP USD"
)

# =======================
# Prediction Button
# =======================
if st.sidebar.button("🚀 Predict HDI"):
    input_features = {
        "Life expectancy": life_expectancy,
        "Expected years of Schooling": expected_schooling,
        "Mean years of Schooling": mean_schooling,
        "GNI($)": gni
    }
    input_df = pd.DataFrame([input_features])

    # Scale inputs
    input_scaled_classification = scaler.transform(input_df)
    input_scaled_regression = scaler_hdi.transform(input_df)

    # Predictions
    prediction_encoded = rf.predict(input_scaled_classification)
    prediction_class = le.inverse_transform(prediction_encoded)
    predicted_hdi_score = regressor.predict(input_scaled_regression)

    # =======================
    # Results Display
    # =======================
    st.subheader("📊 Prediction Result")
    col1, col2 = st.columns(2)

    with col1:
        st.success(f"**HDI Class:** {prediction_class[0]}")

    with col2:
        st.info(f"**HDI Score:** {predicted_hdi_score[0]:.4f}")

else:
    st.warning("👈 Adjust the inputs in the sidebar and click **Predict HDI**.")

# =======================
# Footer
# =======================
st.markdown("---")
st.caption("🔎 HDI: Human Development Index | Model predictions are for **educational/demo use only**.")
