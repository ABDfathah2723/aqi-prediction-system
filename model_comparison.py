import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# ==========================
# CREATE GRAPH FOLDER
# ==========================

os.makedirs("static/graphs", exist_ok=True)

# ==========================
# LOAD DATASET
# ==========================

data = pd.read_csv("cleaned_dataset.csv")

# ==========================
# FEATURES & TARGET
# ==========================

X = data[
    [
        "PM2.5",
        "PM10",
        "NO",
        "NO2",
        "NOx",
        "NH3",
        "CO",
        "SO2",
        "O3"
    ]
]

y = data["AQI_Bucket"]

# ==========================
# LABEL ENCODING
# ==========================

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# MODELS
# ==========================

dt = DecisionTreeClassifier(random_state=42)

knn = KNeighborsClassifier()

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==========================
# TRAIN MODELS
# ==========================

dt.fit(X_train, y_train)
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ==========================
# CROSS VALIDATION
# ==========================

cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores")
print(cv_scores)

print(
    "\nAverage CV Score:",
    cv_scores.mean()
)

# ==========================
# PREDICTIONS
# ==========================

dt_pred = dt.predict(X_test)
knn_pred = knn.predict(X_test)
rf_pred = rf.predict(X_test)

# ==========================
# ACCURACY
# ==========================

dt_acc = accuracy_score(
    y_test,
    dt_pred
)

knn_acc = accuracy_score(
    y_test,
    knn_pred
)

rf_acc = accuracy_score(
    y_test,
    rf_pred
)

print("\nDecision Tree Accuracy:", dt_acc)
print("KNN Accuracy:", knn_acc)
print("Random Forest Accuracy:", rf_acc)

# ==========================
# PRECISION RECALL F1
# ==========================

precision = precision_score(
    y_test,
    rf_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    rf_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    rf_pred,
    average="weighted"
)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# ==========================
# ACCURACY COMPARISON GRAPH
# ==========================

algorithms = [
    "Decision Tree",
    "KNN",
    "Random Forest"
]

accuracy = [
    dt_acc,
    knn_acc,
    rf_acc
]

plt.figure(figsize=(8,5))

bars = plt.bar(
    algorithms,
    accuracy,
    color=[
        "skyblue",
        "orange",
        "green"
    ]
)

plt.title(
    "Algorithm Accuracy Comparison"
)

plt.ylabel("Accuracy")

for bar in bars:

    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{bar.get_height():.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "static/graphs/accuracy_comparison.png"
)

plt.close()

# ==========================
# RANDOM FOREST CONFUSION MATRIX
# ==========================

cm_rf = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "static/graphs/confusion_matrix.png"
)

plt.close()

# ==========================
# DECISION TREE MATRIX
# ==========================

cm_dt = confusion_matrix(
    y_test,
    dt_pred
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title(
    "Decision Tree Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "static/graphs/dt_confusion_matrix.png"
)

plt.close()

# ==========================
# KNN MATRIX
# ==========================

cm_knn = confusion_matrix(
    y_test,
    knn_pred
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm_knn,
    annot=True,
    fmt="d",
    cmap="Oranges"
)

plt.title(
    "KNN Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "static/graphs/knn_confusion_matrix.png"
)

plt.close()

# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = pd.DataFrame({

    "Feature": X.columns,
    "Importance": rf.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title(
    "Feature Importance Ranking"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "static/graphs/feature_importance.png"
)

plt.close()

# ==========================
# AQI CATEGORY COUNT
# ==========================

aqi_counts = data[
    "AQI_Bucket"
].value_counts()

plt.figure(figsize=(8,5))

aqi_counts.plot(
    kind="bar"
)

plt.title(
    "AQI Category Counts"
)

plt.xlabel(
    "AQI Category"
)

plt.ylabel(
    "Count"
)

plt.tight_layout()

plt.savefig(
    "static/graphs/aqi_category_count.png"
)

plt.close()

# ==========================
# AQI DISTRIBUTION PIE
# ==========================

plt.figure(figsize=(8,8))

data["AQI_Bucket"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title(
    "AQI Category Distribution"
)

plt.ylabel("")

plt.savefig(
    "static/graphs/aqi_distribution.png"
)

plt.close()

# ==========================
# PERFORMANCE CSV
# ==========================

performance = pd.DataFrame({

    "Algorithm":[
        "Decision Tree",
        "KNN",
        "Random Forest"
    ],

    "Accuracy":[
        round(dt_acc*100,2),
        round(knn_acc*100,2),
        round(rf_acc*100,2)
    ]
})

performance.to_csv(
    "static/performance_metrics.csv",
    index=False
)

# ==========================
# SAVE BEST MODEL
# ==========================

joblib.dump(
    rf,
    "best_model.pkl"
)

print(
    "\nBest Model Saved Successfully!"
)