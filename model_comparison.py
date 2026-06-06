import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load dataset
data = pd.read_csv("cleaned_dataset.csv")

# Features
X = data[[
    'PM2.5',
    'PM10',
    'NO',
    'NO2',
    'NOx',
    'NH3',
    'CO',
    'SO2',
    'O3'
]]

# Target
y = data['AQI_Bucket']

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()

# Train models
dt.fit(X_train, y_train)
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)

cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=5
)

print("Cross Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())

# Predictions
dt_pred = dt.predict(X_test)
knn_pred = knn.predict(X_test)
rf_pred = rf.predict(X_test)

# Accuracy
dt_acc = accuracy_score(y_test, dt_pred)
knn_acc = accuracy_score(y_test, knn_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("Decision Tree Accuracy:", dt_acc)
print("KNN Accuracy:", knn_acc)
print("Random Forest Accuracy:", rf_acc)

# Compare graph
algorithms = ['Decision Tree', 'KNN', 'Random Forest']
accuracy = [dt_acc, knn_acc, rf_acc]

plt.figure(figsize=(8,5))

plt.bar(algorithms, accuracy)

plt.xlabel("Algorithms")
plt.ylabel("Accuracy")

plt.title("Algorithm Comparison")

plt.savefig('static/graphs/accuracy_comparison.png')
plt.show()

# Confusion Matrrix Graph 
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(8,6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")
plt.savefig('static/graphs/confusion_matrix.png')
plt.show()

# Feature Importance Graph
importance = rf.feature_importances_

features = X.columns

plt.figure(figsize=(10,5))

plt.bar(features, importance)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance Score")

plt.xticks(rotation=45)
plt.savefig("static/graphs/feature_importance.png")
plt.show()

# AQI Distribution Pie Chart
data['AQI_Bucket'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    figsize=(8,8)
)

plt.title("AQI Category Distribution")

plt.ylabel("")
plt.savefig("static/graphs/aqi_distribution.png")
plt.show()

import joblib

joblib.dump(rf, "best_model.pkl")

print("Model Saved Successfully!")