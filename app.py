import streamlit as st
import pandas as pd
import joblib
from risk_scoring import risk_score, risk_level

# Load data and model
df = pd.read_csv("mumbai_house_data_cleaned.csv")
model = joblib.load("random_forest_model.pkl")
features = joblib.load("model_features.pkl")

# UI title
st.title("🏠 Real Estate Price & Risk Prediction")
st.header("Enter Property Details")

# INPUTS 
region = st.selectbox("Region", sorted(df["region"].unique()))
bhk = st.selectbox("BHK", sorted(df["bhk"].unique()))
ptype = st.selectbox("Type", sorted(df["type"].unique()))
area = st.number_input("Area (sq ft)", min_value=100, max_value=16000, value=100)

# Predict Button 
if st.button("Predict Price and Risk"):
    # Find matching row for risk values
    match_row = df[(df["region"] == region) &
                   (df["bhk"] == bhk) &
                   (df["type"] == ptype)]

         
    # If no match found → stop
    if match_row.empty:
        st.error("No matching data found in dataset.")
        st.stop()

    # Get risk factors from the first matching row
    row = match_row.iloc[0]
    roi = row['expected_roi(%)']
    demand = row['demand_indicator']
    volatility = row['market_volatitlity_score']
    liquidity = row['property_liquidity_index']
        

    # Encode inputs for model
    region_encoded = pd.factorize(df["region"])[0][df["region"] == region][0]
    type_encoded = pd.factorize(df["type"])[0][df["type"] == ptype][0]

    # Create input for model
    input_data = pd.DataFrame([[
        bhk,
        type_encoded,
        0,        # locality (dummy)
        area,
        region_encoded,
        0,        # status (dummy)
        0,        # age (dummy)
        demand,
        volatility,
        liquidity
    ]], columns=features)

    # Predict price
    predicted_price = model.predict(input_data)[0]
    
    
    # OUTPUT
    st.subheader("Estimated Price")
    st.success(f"₹ {predicted_price:.2f} Lakhs")

    st.subheader("Risk Report")
    st.markdown(f"- **Expected ROI**: {roi}%")
    st.markdown(f"- **Demand Indicator**: {demand}/10")
    st.markdown(f"- **Market Volatility**: {volatility}/10")
    st.markdown(f"- **Liquidity Index**: {liquidity}/10")

    # Risk Score
    risk_row = {
        'expected_roi(%)': roi,
        'demand_indicator': demand,
        'market_volatitlity_score': volatility,
        'property_liquidity_index': liquidity
    }
    score = risk_score(pd.Series(risk_row))
    level = risk_level(score)

    st.subheader("Total Risk Score")
    st.info(f"{score} / 10 — {level}")

    st.subheader("Recommendation")
    if level == "Low Risk":
        st.success("Safe and stable place to invest or live.")
    elif level == "Moderate Risk":
        st.warning("Good for short - to mid-term stay or investment")
    else:
        st.error("High risk area. Consider other safer locations for long-term investment.")



