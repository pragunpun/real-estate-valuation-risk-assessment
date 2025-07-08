import pandas as pd

# Normalize ROI to 0–10 scale using actual data range
def normalize_roi(roi, min_roi=3.31, max_roi=15.0):
    return 10 * (roi - min_roi) / (max_roi - min_roi)

# Total risk score calculation
def risk_score(row, min_roi=3.31, max_roi=15.0):
    roi_norm = normalize_roi(row['expected_roi(%)'], min_roi, max_roi) # Scale 0–10
    roi_risk = 10 - roi_norm # High ROI → Low risk
    demand_risk = 10 - row['demand_indicator'] # High demand → Low risk
    volatility_risk = row['market_volatitlity_score'] # High volatility → High risk
    liquidity_risk = 10 - row['property_liquidity_index'] # High liquidity → Low risk

    total_risk = (roi_risk + demand_risk + volatility_risk + liquidity_risk) / 4
    return round(total_risk, 2)

# Risk category based on total risk score
def risk_level(score):
    if score < 4:
        return "Low Risk"
    elif score <= 7:
        return "Moderate Risk"
    else:
        return "High Risk"


