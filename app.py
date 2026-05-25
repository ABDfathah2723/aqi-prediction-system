from flask import Flask, render_template, request
import numpy as np
import joblib
import sqlite3

app = Flask(__name__)

# Load trained model
model = joblib.load("best_model.pkl")


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