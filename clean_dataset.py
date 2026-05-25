import pandas as pd

# Load dataset
data = pd.read_csv("dataset.csv")

# Select needed columns
data = data[[
    'PM2.5',
    'PM10',
    'NO',
    'NO2',
    'NOx',
    'NH3',
    'CO',
    'SO2',
    'O3',
    'AQI_Bucket'
]]

# Remove missing values
data = data.dropna()

# Save cleaned dataset
data.to_csv("cleaned_dataset.csv", index=False)

print(data.head())

print("\nDataset Cleaned Successfully!")