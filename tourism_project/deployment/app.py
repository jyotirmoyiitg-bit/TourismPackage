"""Streamlit app: predict whether a customer will purchase the Wellness Tourism Package."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "best_tourism_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


artifact = load_model()
model, threshold = artifact["model"], artifact["threshold"]

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳")
st.title("🧳 Wellness Tourism Package Predictor")
st.write("Enter the customer details to predict whether they are likely to purchase the package.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer details")
    age = st.number_input("Age", 18, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly income", 1000, 100000, 20000, step=500)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    passport = st.selectbox("Holds a valid passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])
    persons = st.number_input("Number of people visiting", 1, 10, 3)
    children = st.number_input("Children (under 5) visiting", 0, 5, 0)
    trips = st.number_input("Average trips per year", 0, 25, 3)
    star = st.selectbox("Preferred property star", [3.0, 4.0, 5.0])
with col2:
    st.subheader("Interaction details")
    contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration = st.number_input("Duration of pitch (minutes)", 1, 150, 15)
    followups = st.number_input("Number of follow-ups", 0, 10, 3)
    satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)

# Collect the inputs into a dataframe with the same columns used in training
input_df = pd.DataFrame([{
    "Age": age, "TypeofContact": contact, "CityTier": city_tier, "DurationOfPitch": duration,
    "Occupation": occupation, "Gender": gender, "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups, "ProductPitched": product, "PreferredPropertyStar": star,
    "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": int(passport == "Yes"),
    "PitchSatisfactionScore": satisfaction, "OwnCar": int(own_car == "Yes"),
    "NumberOfChildrenVisiting": children, "Designation": designation, "MonthlyIncome": income,
}])

if st.button("Predict", type="primary"):
    proba = model.predict_proba(input_df)[0, 1]
    if proba >= threshold:
        st.success(f"Likely to purchase the Wellness Package (probability {proba:.1%})")
    else:
        st.warning(f"Unlikely to purchase the Wellness Package (probability {proba:.1%})")
    with st.expander("Input data"):
        st.dataframe(input_df)
