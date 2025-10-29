import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🏠 Airbnb Price Predictor", layout="centered")
st.markdown(
    """
    <h1 style='text-align:center; color:#ff4b4b;'>🏠 Airbnb Price Predictor</h1>
    <p style='text-align:center; color:gray;'>Estimate nightly prices based on listing details</p>
    <hr>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    pre = joblib.load("preprocessor.pkl")
    return model, pre

model, preprocessor = load_artifacts()

st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "This app predicts the estimated nightly price of an Airbnb listing using ML trained on U.S. data.\n\n"
    "Model: **XGBoost Regressor**\nData: **Inside Airbnb (U.S.)**"
)

# popular cities from dataset
city_list = [
    "Los Angeles", "New York City", "San Francisco", "Austin", "Chicago",
    "Miami", "Seattle", "Boston", "Portland", "Denver", "Las Vegas", "Nashville"
]

st.markdown("### 📝 Fill in listing details")

with st.form("airbnb_form"):
    with st.expander("🏙️ Basic Info", expanded=True):
        city = st.selectbox("City", city_list, index=0)
        neighbourhood = st.text_input("Neighbourhood", "Hollywood")
        room_type = st.selectbox("Room Type", ["Entire home/apt","Private room","Shared room"])
        property_type = st.selectbox("Property Type", ["Apartment","House","Condominium","Loft"])

    with st.expander("🛏️ Space Details"):
        accommodates = st.number_input("Accommodates", 1, 16, 2)
        bedrooms = st.number_input("Bedrooms", 0, 10, 1)
        bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 1.0, step=0.5)
        minimum_nights = st.number_input("Minimum Nights", 1, 30, 2)
        availability_365 = st.number_input("Availability (days/year)", 0, 365, 120)

    with st.expander("⭐ Host & Reviews"):
        number_of_reviews = st.number_input("Number of Reviews", 0, 5000, 10)
        review_scores_rating = st.slider("Review Score (0–100)", 0, 100, 90)
        host_is_superhost = st.selectbox("Host is Superhost?", ["Yes","No"])

    with st.expander("🧰 Amenities"):
        cols = st.columns(3)
        with cols[0]:
            amen_wifi = st.checkbox("WiFi")
            amen_kitchen = st.checkbox("Kitchen")
            amen_parking = st.checkbox("Parking")
        with cols[1]:
            amen_air_conditioning = st.checkbox("Air Conditioning")
            amen_heating = st.checkbox("Heating")
            amen_tv = st.checkbox("TV")
        with cols[2]:
            amen_pool = st.checkbox("Pool")
            amen_dryer = st.checkbox("Dryer")
            amen_washer = st.checkbox("Washer")

    submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

if submitted:
    row = {
        "city": city,
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "property_type": property_type,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "review_scores_rating": review_scores_rating,
        "host_is_superhost": 1 if host_is_superhost=="Yes" else 0,
        "availability_365": availability_365,

        # geo defaults per city
        "latitude": 34.0522,
        "longitude": -118.2437,

        # amenities
        "amen_wifi": int(amen_wifi),
        "amen_kitchen": int(amen_kitchen),
        "amen_parking": int(amen_parking),
        "amen_air_conditioning": int(amen_air_conditioning),
        "amen_heating": int(amen_heating),
        "amen_tv": int(amen_tv),
        "amen_pool": int(amen_pool),
        "amen_dryer": int(amen_dryer),
        "amen_washer": int(amen_washer),

        "log_price": 0
    }

    X = pd.DataFrame([row])

    # ensure all required columns exist
    missing_cols = set(preprocessor.feature_names_in_) - set(X.columns)
    for c in missing_cols:
        X[c] = 0
    X = X[preprocessor.feature_names_in_]

    X_proc = preprocessor.transform(X)
    price = float(model.predict(X_proc)[0])

    st.success(f"💰 Estimated Nightly Price: **${price:,.2f}**")
    st.caption("_Note: estimates are based on 2020 US Airbnb data and may vary with real market conditions._")
