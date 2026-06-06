import sqlite3

def create_database():

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        PM25 REAL NOT NULL,
        PM10 REAL NOT NULL,
        NO REAL NOT NULL,
        NO2 REAL NOT NULL,
        NOx REAL NOT NULL,
        NH3 REAL NOT NULL,
        CO REAL NOT NULL,
        SO2 REAL NOT NULL,
        O3 REAL NOT NULL,

        Prediction TEXT NOT NULL,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

    print("Database Created Successfully!")

if __name__ == "__main__":
    create_database()