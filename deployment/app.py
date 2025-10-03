# step 1: load model from Hugging Face hub
from huggingface_hub import hf_hub_download
import joblib

# Download model from HF hub
model_path = hf_hub_download(
    repo_id="sahithisaranya/tourismxgb_model",
    filename="best_pipeline.joblib"
)

# Load model
model = joblib.load(model_path)

import streamlit as st
import pandas as pd

st.title("Wellness Tourism Package Predictor 🌍")

# Collect user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income", min_value=0, step=1000, value=50000)
city_tier = st.selectbox("City Tier", [1, 2, 3])
gender = st.selectbox("Gender", ["Male", "Female"])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
passport = st.selectbox("Has Passport?", [0, 1])
own_car = st.selectbox("Own Car?", [0, 1])
num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=1)
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# Build dataframe
input_dict = {
    "Age": [age],
    "MonthlyIncome": [income],
    "CityTier": [city_tier],
    "Gender": [gender],
    "Occupation": [occupation],
    "MaritalStatus": [marital_status],
    "Passport": [passport],
    "OwnCar": [own_car],
    "NumberOfTrips": [num_trips],
    "PitchSatisfactionScore": [pitch_score],
}

input_df = pd.DataFrame(input_dict)

st.write("### Preview of Input Data")
st.write(input_df)

