from flask import Flask, render_template, request
import os
import requests
import sqlite3
import joblib
import pandas as pd

app = Flask(__name__)

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("best_model.pkl")

# ==========================
# API KEY
# ==========================

API_KEY = os.getenv("API_KEY")

# ==========================
# HOME PAGE
# ==========================

@app.route('/')
def home():
    return render_template('index.html')

# ==========================
# DASHBOARD PAGE
# ==========================

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# ==========================
# DATASET PAGE
# ==========================

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

# ==========================
# ABOUT PAGE
# ==========================

@app.route('/about')
def about():
    return render_template('about.html')

# ==========================
# PERFORMANCE PAGE
# ==========================

@app.route('/performance')
def performance():
    return render_template('performance.html')

# ==========================
# HISTORY PAGE
# ==========================

@app.route('/history')
def history():

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM predictions
    ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    cursor.execute("""
    SELECT COUNT(*) FROM predictions
    """)

    total_predictions = cursor.fetchone()[0]

    conn.close()

    return render_template(
        'history.html',
        rows=rows,
        total_predictions=total_predictions
    )

# ==========================
# LIVE AQI PAGE
# ==========================

@app.route('/live/<city>')
def live(city):

    api_key = "9529e6bfcb617d03055e4d73a9ce3132"

    cities = {
        "delhi": (28.7041, 77.1025),
        "bangalore": (12.9716, 77.5946),
        "chennai": (13.0827, 80.2707),
        "mumbai": (19.0760, 72.8777)
    }

    if city not in cities:
        return "City not found"

    lat, lon = cities[city]

    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"

    response = requests.get(url)

    data = response.json()

    pollution = data['list'][0]['components']

    return render_template(
        'live.html',
        city=city,
        pollution=pollution
    )

# ==========================
# CITY AQI SEARCH
# ==========================

@app.route('/aqi', methods=['POST'])
def aqi():
    api_key = "9529e6bfcb617d03055e4d73a9ce3132"
    city = request.form['city']

    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"

    geo_response = requests.get(geo_url)

    geo_data = geo_response.json()

    if len(geo_data) == 0:

        return render_template(
            "index.html",
            prediction_text="City not found!"
        )

    lat = geo_data[0]['lat']
    lon = geo_data[0]['lon']

    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"

    aqi_response = requests.get(aqi_url)

    aqi_data = aqi_response.json()

    aqi_index = aqi_data['list'][0]['main']['aqi']

    pm25 = aqi_data['list'][0]['components']['pm2_5']
    pm10 = aqi_data['list'][0]['components']['pm10']

    if aqi_index == 1:
        category = "Good 😊"

    elif aqi_index == 2:
        category = "Fair 🙂"

    elif aqi_index == 3:
        category = "Moderate 😐"

    elif aqi_index == 4:
        category = "Poor 😷"

    else:
        category = "Very Poor ☠"

    return render_template(
        'index.html',
        city=city,
        aqi=aqi_index,
        pm25=pm25,
        pm10=pm10,
        category=category
    )

# ==========================
# PREDICTION ROUTE
# ==========================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        PM25 = float(request.form['PM25'])
        PM10 = float(request.form['PM10'])
        NO = float(request.form['NO'])
        NO2 = float(request.form['NO2'])
        NOx = float(request.form['NOx'])
        NH3 = float(request.form['NH3'])
        CO = float(request.form['CO'])
        SO2 = float(request.form['SO2'])
        O3 = float(request.form['O3'])

    except ValueError:

        return render_template(
            'index.html',
            prediction_text="Invalid Input! Enter numbers only."
        )

    features = pd.DataFrame([{

        "PM2.5": PM25,
        "PM10": PM10,
        "NO": NO,
        "NO2": NO2,
        "NOx": NOx,
        "NH3": NH3,
        "CO": CO,
        "SO2": SO2,
        "O3": O3

    }])

    prediction = model.predict(features)[0]

    aqi_labels = {

        0: "Good 😊",
        1: "Moderate 😐",
        2: "Poor 😷",
        3: "Very Poor 🤢",
        4: "Severe ☠️"

    }

    prediction = aqi_labels.get(
        prediction,
        str(prediction)
    )

    confidence = round(
        max(model.predict_proba(features)[0]) * 100,
        2
    )

    category = prediction

    if "Good" in category:

        color = "#00c853"
        advice = "Air quality is good. Safe for outdoor activities."

    elif "Moderate" in category:

        color = "#ffd600"
        advice = "Air quality is acceptable."

    elif "Poor" in category:

        color = "#ff9100"
        advice = "Avoid outdoor exercise."

    elif "Very Poor" in category:

        color = "#ff3d00"
        advice = "Limit outdoor activities."

    else:

        color = "#b71c1c"
        advice = "Stay indoors and wear protection."

    # ==========================
    # SAVE TO DATABASE
    # ==========================

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions(
        PM25,
        PM10,
        NO,
        NO2,
        NOx,
        NH3,
        CO,
        SO2,
        O3,
        Prediction
    )

    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,

    (
        PM25,
        PM10,
        NO,
        NO2,
        NOx,
        NH3,
        CO,
        SO2,
        O3,
        prediction
    ))

    conn.commit()
    conn.close()

    return render_template(

        'index.html',

        prediction_text=prediction,
        category=category,
        color=color,
        advice=advice,
        confidence=confidence

    )

# ==========================
# RUN APP
# ==========================

if __name__ == "__main__":
    app.run(debug=True)
