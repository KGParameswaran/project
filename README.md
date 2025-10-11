# 🌍 HDI Prediction (Machine Learning + Streamlit Dashboard)

## Project Objective
The project aims to predict the **Human Development Index (HDI)** class and score of a country using socio-economic indicators.  
This project demonstrates the use of **machine learning models** and a **Streamlit web app** to make predictions in a user-friendly, interactive interface.  

---

## Datasets / Models Used
The project is powered by pre-trained machine learning models and preprocessing tools:

- **Random Forest Classifier** – predicts HDI class (Low, Medium, High, Very High)  
- **Random Forest Regressor** – predicts HDI score (0–1 scale)  
- **Standard Scalers** – normalize input features  
- **Label Encoder** – encodes/decodes HDI classes  

Input Features used for prediction:
- 📈 Life Expectancy (years)  
- 🎓 Expected Years of Schooling  
- 🎓 Mean Years of Schooling  
- 💰 Gross National Income (GNI) per capita  

---

## Dashboard (Streamlit App)
The interactive Streamlit app allows users to:
- Input values for socio-economic indicators.  
- Predict **HDI Class** and **HDI Score** in real time.  
- View explanations of **What HDI is**.  

👉 The app is deployed on **Streamlit Community Cloud** for online access.  

---

## Conclusion
I developed a comprehensive and interactive **Streamlit dashboard** to predict the **Human Development Index (HDI)** using machine learning models.  
This included data preprocessing, training classification and regression models, and designing an engaging user interface.  

The dashboard provides insights such as:  
- HDI Class categories (Low, Medium, High, Very High)  
- HDI Score (0–1 range)  
- How life expectancy, education, and income impact HDI  

⚠️ *Note: Predictions are for educational/demo purposes only and not for official use.*  
