import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# Download model from HF hub
model_path = hf_hub_download(
    repo_id="sahithisaranya/tourismxgb_model",
    filename="best_pipeline.joblib",
    token=os.environ.get("HF_TOKEN")
)

# Load model
model = joblib.load(model_path)

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

# Create a dictionary with the collected inputs
input_data = {
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
    # Add other columns that the model expects, even if not used in the UI
    # These should have default or placeholder values
    "TypeofContact": ["Self Enquiry"], # Example default
    "NumberOfPersonVisiting": [1], # Example default
    "NumberOfFollowups": [1.0], # Example default
    "ProductPitched": ["Basic"], # Example default
    "PreferredPropertyStar": [3.0], # Example default
    "NumberOfChildrenVisiting": [0.0], # Example default
    "Designation": ["Executive"], # Example default
}

# Ensure the input DataFrame has the same columns and order as the training data
# This is crucial for the preprocessor
# Load a small sample of the training data to get the column order and types
try:
    dataset_train_sample = load_dataset("sahithisaranya/tourism_dataset_processed", split="train[:1]")
    train_cols = dataset_train_sample.features.keys()
    # Remove the target variable if it exists
    if "ProdTaken" in train_cols:
        train_cols = [col for col in train_cols if col != "ProdTaken"]
    input_df = pd.DataFrame(input_data)
    # Reindex columns to match the training data order
    input_df = input_df.reindex(columns=train_cols, fill_value=0) # Use fill_value=0 or appropriate default for missing columns
except Exception as e:
    st.error(f"Error loading training data sample for column reindexing: {e}")
    input_df = pd.DataFrame(input_data) # Fallback to using input_data as is


st.write("### Preview of Input Data")
st.write(input_df)

# Make prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]

        st.write("### Prediction")
        if prediction[0] == 1:
            st.success(f"The model predicts that the customer is likely to purchase the Wellness Tourism Package (Probability: {prediction_proba[0]:.2f}).")
        else:
            st.info(f"The model predicts that the customer is unlikely to purchase the Wellness Tourism Package (Probability: {prediction_proba[0]:.2f}).")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
