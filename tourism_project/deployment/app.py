import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="", filename="Tourismpkg_prediction_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism package Prediction")
st.write("""
This application predicts the expected **ad revenue** of a Play Store application
based on its characteristics such as category, installs, active users, and screen time.
Please enter the app details below to get a revenue prediction.
""")

# User input

TypeofContact = st.selectbox("Company Invited or Self Enquiry", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("The City category", ["Tier1", "Tier2", "Tier3"])
Occupation = st.selectbox("Customer Occupation", ["Salaried", "Freelancer"])
Gender = st.selectbox("Gender of the customer", ["Male", "Female"])
Martial_Status = st.selectbox("Martial status of the customer", ["Single", "Married","Divorced"])
Designation = st.selectbox("Designation of the customer", ["Executive", "Managerial","Professional","Self-Employed"])

Age = st.number_input("Customer Age", min_value=1.0, max_value=120.0, value=18.0, step=0.1)
NumberOfPersonVisiting = st.number_input("Total num of people accompanying the customer", min_value=1.0, max_value=1000.0, value=1.0, step=1)
PreferredPropertyStar = st.number_input("Preferred hotel rating y the customer",min_value=1.0, max_value=10.0, value=1.0, step=1)
NumberOfTrips = st.number_input("Avg num of trips the customer takes annually",min_value=1.0, max_value=10.0, value=1.0, step=1)
Passport = st.number_input("Yes-1 or No-0",min_value=0, max_value=1, value=0)
OwnCar = st.number_input("Yes-1 or No-0",min_value=0, max_value=1, value=0)
NumberOfChildrenVisiting = st.number_input("Total num of childern (<5 years) accompanying the customer", min_value=1.0, max_value=5, value=0, step=1)
MonthlyIncome = st.number_input("Gross monthly income of the customer", value=10000)



# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'Martial_Status': Martial_Status,
    'Designation': Designation,
    'Age': Age,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Yes" if prediction ==1 else "No"
    st.subheader("Prediction Result:")
    st.success(f": Modle predict :: Customer Prod taken **{result}**")
