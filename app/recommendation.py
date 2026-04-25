import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.utils import calculate_accessibility

scaler = joblib.load('models/scaler.pkl')
features = joblib.load('models/features.pkl')

df = pd.read_csv('data/combined_final_dataset.csv')


def recommend_properties(user_input, budget, top_n=5):

    # ----------------------------
    # Ensure column exists
    # ----------------------------
    if 'AccessibilityScore' not in df.columns:
        df['AccessibilityScore'] = 0.5

    # ----------------------------
    # USER INPUT
    # ----------------------------
    user_area = user_input.get("CarpetArea_sqft", 0)
    user_bhk = user_input.get("BHK", 0)

    # ----------------------------
    # 🔥 FILTER 1: AREA
    # ----------------------------
    df_filtered = df[
        (df["CarpetArea_sqft"] >= 0.8 * user_area) &
        (df["CarpetArea_sqft"] <= 1.2 * user_area)
    ]

    # ----------------------------
    # 🔥 FILTER 2: BHK
    # ----------------------------
    df_filtered = df_filtered[
        df_filtered["BHK"].between(user_bhk - 1, user_bhk + 1)
    ]

    # ----------------------------
    # 🔥 FILTER 3: BUDGET
    # ----------------------------
    df_filtered = df_filtered[
        df_filtered["Price_INR"] <= 1.2 * budget
    ]

    # ----------------------------
    # FALLBACK (VERY IMPORTANT)
    # ----------------------------
    if len(df_filtered) < 10:
        df_filtered = df.copy()

    # ----------------------------
    # Data for model
    # ----------------------------
    data = df_filtered[features]

    user_df = pd.DataFrame([user_input])
    user_df = user_df.reindex(columns=features, fill_value=0)

    # ----------------------------
    # Scaling
    # ----------------------------
    data_scaled = scaler.transform(data)
    user_scaled = scaler.transform(user_df)

    # ----------------------------
    # Feature Weights
    # ----------------------------
    weights = np.ones(len(features))

    for i, f in enumerate(features):
        if f == "BHK":
            weights[i] = 3
        elif "Area" in f:
            weights[i] = 3
        elif f == "Bathrooms":
            weights[i] = 2
        elif f == "AccessibilityScore":
            weights[i] = 2

    data_scaled = data_scaled * weights
    user_scaled = user_scaled * weights

    # ----------------------------
    # Similarity
    # ----------------------------
    sim = cosine_similarity(user_scaled, data_scaled)[0]

    scores = sorted(list(enumerate(sim)), key=lambda x: x[1], reverse=True)
    top = scores[:top_n]

    indices = [i[0] for i in top]
    raw_scores = [i[1] for i in top]

    # ----------------------------
    # Normalize match %
    # ----------------------------
    min_s, max_s = min(raw_scores), max(raw_scores)

    match_percents = [
        round((s - min_s) / (max_s - min_s + 1e-5) * 100)
        for s in raw_scores
    ]

    # ----------------------------
    # Get final rows
    # ----------------------------
    result = df_filtered.iloc[indices].copy()

    # ----------------------------
    # Accessibility (only top rows)
    # ----------------------------
    result['AccessibilityScore'] = result.apply(
        lambda row: calculate_accessibility(
            row['City'], row['Latitude'], row['Longitude']
        ),
        axis=1
    )

    result['MatchPercent'] = match_percents

    # ----------------------------
    # Affordability
    # ----------------------------
    result['AffordabilityScore'] = budget / result['Price_INR']

    def label(x):
        if x >= 1:
            return "Affordable ✅"
        elif x >= 0.8:
            return "Slightly Expensive ⚠️"
        else:
            return "Expensive ❌"

    result['AffordabilityLabel'] = result['AffordabilityScore'].apply(label)

    # ----------------------------
    # Final output
    # ----------------------------
    result = result[
    [
        'BHK',
        'Bathrooms',
        'Balconies',
        'CarpetArea_sqft',
        'BuiltUpArea_sqft',
        'SuperBuiltUpArea_sqft',
        'Floor',
        'TotalFloors',
        'Furnishing',
        'Parking',
        'BuildingType',
        'Facing',
        'AmenitiesCount',
        'Price_INR',
        'Latitude',
        'Longitude',
        'AccessibilityScore',
        'MatchPercent',
        'AffordabilityLabel',
        'IsRERARegistered'
    ]
]
    return result.to_dict(orient='records') 