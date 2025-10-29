import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Airbnb Price Predictor", layout="centered")
st.title("🏠 Airbnb Price Predictor")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    pre = joblib.load("preprocessor.pkl")
    return model, pre

model, preprocessor = load_artifacts()

st.markdown("Fill in listing details and click **Predict**.")

with st.form("form"):
    city = st.text_input("City", "Los Angeles")
    neighbourhood = st.text_input("Neighbourhood", "Hollywood")
    room_type = st.selectbox("Room Type", ["Entire home/apt","Private room","Shared room"])
    property_type = st.selectbox("Property Type", ["Apartment","House","Condominium","Loft"])
    accommodates = st.number_input("Accommodates", 1, 16, 2)
    bedrooms = st.number_input("Bedrooms", 0, 10, 1)
    bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 1.0, step=0.5)
    number_of_reviews = st.number_input("Number of Reviews", 0, 5000, 10)
    review_scores_rating = st.number_input("Review Score (0–100)", 0, 100, 90)
    host_is_superhost = st.selectbox("Host is Superhost?", ["Yes","No"])
    availability_365 = st.number_input("Availability (days/year)", 0, 365, 120)
    amen_wifi = st.checkbox("WiFi")
    amen_kitchen = st.checkbox("Kitchen")
    amen_parking = st.checkbox("Parking")
    amen_air_conditioning = st.checkbox("Air Conditioning")
    submitted = st.form_submit_button("🔮 Predict Price")

if submitted:
    row = {
        "city": city,
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "property_type": property_type,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "number_of_reviews": number_of_reviews,
        "review_scores_rating": review_scores_rating,
        "host_is_superhost": 1 if host_is_superhost=="Yes" else 0,
        "availability_365": availability_365,
        "amen_wifi": int(amen_wifi),
        "amen_kitchen": int(amen_kitchen),
        "amen_parking": int(amen_parking),
        "amen_air_conditioning": int(amen_air_conditioning),
    }
    X = pd.DataFrame([row])

    # transform + predict
    X_proc = preprocessor.transform(X)
    price = float(model.predict(X_proc)[0])

    st.success(f"💰 estimated nightly price: **${price:,.2f}**")
    st.caption("note: this is a statistical estimate; actual prices vary by seasonality, fees, and market shifts.")
