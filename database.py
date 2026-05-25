import sqlite3

conn = sqlite3.connect("predictions.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    PM25 REAL,
    PM10 REAL,
    NO REAL,
    NO2 REAL,
    NOx REAL,
    NH3 REAL,
    CO REAL,
    SO2 REAL,
    O3 REAL,

    Prediction TEXT
)
""")

conn.commit()

print("Database Created Successfully!")