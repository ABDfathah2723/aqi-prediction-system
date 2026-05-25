from flask import Flask, render_template, request
import os
import requests
import numpy as np
import joblib
import sqlite3

app = Flask(__name__)

# Load trained model
model = joblib.load("best_model.pkl")

# LOAD API KEY
API_KEY = os.getenv("API_KEY")

# HOME PAGE
@app.route('/')

def home():

    return render_template('index.html')


# DASHBOARD PAGE
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# HISTORY PAGE
@app.route('/history')
def history():

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions")

    rows = cursor.fetchall()

    return render_template(
        'history.html',
        rows=rows
    )
@app.route('/aqi', methods=['POST'])

def aqi():

    city = request.form['city']

    # STEP 1 → GET CITY COORDINATES

    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"

    geo_response = requests.get(geo_url)

    geo_data = geo_response.json()

    lat = geo_data[0]['lat']

    lon = geo_data[0]['lon']

    # STEP 2 → GET AQI DATA

    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    aqi_response = requests.get(aqi_url)

    aqi_data = aqi_response.json()

    # AQI VALUE

    aqi_index = aqi_data['list'][0]['main']['aqi']

    # POLLUTION COMPONENTS

    pm25 = aqi_data['list'][0]['components']['pm2_5']

    pm10 = aqi_data['list'][0]['components']['pm10']

    # AQI CATEGORY

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

# ADD LIVE AQI ROUTE
@app.route('/live/<city>')
def live(city):

    api_key = "9529e6bfcb617d03055e4d73a9ce3132"

    cities = {
        "delhi": (28.7041, 77.1025),
        "bangalore": (12.9716, 77.5946),
        "chennai": (13.0827, 80.2707),
        "mumbai": (19.0760, 72.8777)
    }

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

# PERFORMANCE ANALYSIS PAGE
@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/about')
def about():
    return render_template('about.html')

# PREDICTION
@app.route('/predict', methods=['POST'])
def predict():

    PM25 = float(request.form['PM25'])
    PM10 = float(request.form['PM10'])
    NO = float(request.form['NO'])
    NO2 = float(request.form['NO2'])
    NOx = float(request.form['NOx'])
    NH3 = float(request.form['NH3'])
    CO = float(request.form['CO'])
    SO2 = float(request.form['SO2'])
    O3 = float(request.form['O3'])

    features = np.array([[
        PM25,
        PM10,
        NO,
        NO2,
        NOx,
        NH3,
        CO,
        SO2,
        O3
    ]])

    prediction = model.predict(features)

    categories = [
        "Good 😊",
        "Moderate 😐",
        "Poor 😷",
        "Very Poor 🥴",
        "Severe ☠"
    ]

    result = categories[prediction[0]]

    color = ""
    advice = ""

    if "Good" in result:

        advice = "Air quality is good 😊"

        color = "#28a745"

    elif "Moderate" in result:

        advice = "Air quality is acceptable 😐"

        color = "#ffc107"

    elif "Poor" in result:

        advice = "Sensitive people should avoid outdoor activities 😷"

        color = "#fd7e14"

    elif "Very Poor" in result:

        advice = "Wear mask while going outside 🥴"

        color = "#dc3545"

    else:

        advice = "Avoid outdoor activities ☠"

        color = "#7b0000"

    # SAVE TO DATABASE

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (
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
        result
    ))

    conn.commit()

    return render_template(

    'index.html',

    prediction_text=result,

    advice=advice,

    color=color
)


if __name__ == "__main__":
    app.run(debug=True)