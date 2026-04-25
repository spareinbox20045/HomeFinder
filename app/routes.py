from flask import Blueprint, render_template, request
import pandas as pd

from app.prediction import predict_price
from app.recommendation import recommend_properties
from app.utils import calculate_accessibility

main = Blueprint('main', __name__)

# load dataset
df = pd.read_csv('data/combined_final_dataset.csv')

cities = sorted(df['City'].dropna().unique())


@main.route('/', methods=['GET'])
def home():
    return render_template('index.html', cities=cities)


@main.route('/predict', methods=['POST'])
def predict():

    # ----------------------------
    # 1. Get form data
    # ----------------------------
    city = request.form.get('city')
    bhk = int(request.form.get('bhk'))
    bathrooms = int(request.form.get('bathrooms'))
    balconies = int(request.form.get('balconies'))
    area = float(request.form.get('area'))
    age = int(request.form.get('age'))
    amenities = int(request.form.get('amenities'))
    budget = float(request.form.get('budget')) * 100000
    sort_by = request.form.get('sort', 'match')

    # ----------------------------
    # 2. City → coordinates
    # ----------------------------
    city_data = df[df['City'] == city]

    if city_data.empty:
        lat, lon = 28.6139, 77.2090
    else:
        lat = city_data['Latitude'].mean()
        lon = city_data['Longitude'].mean()

    # ----------------------------
    # 3. Accessibility
    # ----------------------------
    access_score = calculate_accessibility(city,lat, lon)
    # ----------------------------
    # 4. Input for model
    # ----------------------------
    user_input = {
        'BHK': bhk,
        'Bathrooms': bathrooms,
        'Balconies': balconies,
        'CarpetArea_sqft': area,
        'AgeYears': age,
        'AmenitiesCount': amenities,
        'Latitude': lat,
        'Longitude': lon,
        'AccessibilityScore': access_score
    }

    # ----------------------------
    # 5. Prediction
    # ----------------------------
    price_data = predict_price(user_input)
    predicted_price = price_data["predicted_price"]

    # ----------------------------
    # 6. Affordability
    # ----------------------------
    affordability_score = budget / predicted_price

    if affordability_score >= 1:
        affordability_label = "Affordable ✅"
    elif affordability_score >= 0.8:
        affordability_label = "Slightly Expensive ⚠️"
    else:
        affordability_label = "Expensive ❌"

    # ----------------------------
    # 7. Recommendations
    # ----------------------------
    recommendations = recommend_properties(user_input, budget)
    # ----------------------------
    # 8. Sorting
    # ----------------------------
    if sort_by == "area":
        recommendations = sorted(
            recommendations,
            key=lambda x: x['CarpetArea_sqft'],
            reverse=True
        )
    else:
        recommendations = sorted(
            recommendations,
            key=lambda x: x['MatchPercent'],
            reverse=True
        )

    # ----------------------------
    # 9. Render
    # ----------------------------
    return render_template(
        'results.html',
        price=price_data,
        results=recommendations,
        city=city,
        affordability=round(affordability_score, 2),
        affordability_label=affordability_label
    )