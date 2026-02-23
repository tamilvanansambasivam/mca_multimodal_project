import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

import os

# Create "models" folder if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")
    print("Created 'models' folder.")
else:
    print("'models' folder already exists.")

df = pd.read_csv("dataset.csv")

# -----------------------------
# Model 1: Force Prediction
# -----------------------------
X = df[["S1","S2","S3","S4","S5"]]
y_force = df["Force"]

X_train, X_test, yf_train, yf_test = train_test_split(
    X, y_force, test_size=0.2, random_state=42
)

force_model = RandomForestRegressor(n_estimators=100)
force_model.fit(X_train, yf_train)

pred_force = force_model.predict(X_test)
rmse_force = np.sqrt(mean_squared_error(yf_test, pred_force))

print("Force RMSE:", rmse_force)

joblib.dump(force_model, "models/force_model.pkl")

# -----------------------------
# Cascaded Features
# -----------------------------
df["PredictedForce"] = force_model.predict(X)

X_cascade = df[["S1","S2","S3","S4","S5","PredictedForce"]]

# -----------------------------
# Model 2: Orientation
# -----------------------------
y_orient = df["Orientation"]

X_train, X_test, yo_train, yo_test = train_test_split(
    X_cascade, y_orient, test_size=0.2, random_state=42
)

orient_model = RandomForestClassifier(n_estimators=100)
orient_model.fit(X_train, yo_train)

pred_orient = orient_model.predict(X_test)
acc_orient = accuracy_score(yo_test, pred_orient)

print("Orientation Accuracy:", acc_orient)

joblib.dump(orient_model, "models/orientation_model.pkl")

# -----------------------------
# Model 3: Curvature
# -----------------------------
y_curv = df["Curvature"]

X_train, X_test, yc_train, yc_test = train_test_split(
    X_cascade, y_curv, test_size=0.2, random_state=42
)

curv_model = RandomForestRegressor(n_estimators=100)
curv_model.fit(X_train, yc_train)

pred_curv = curv_model.predict(X_test)
rmse_curv = np.sqrt(mean_squared_error(yc_test, pred_curv))

print("Curvature RMSE:", rmse_curv)

joblib.dump(curv_model, "models/curvature_model.pkl")

print("All models trained and saved successfully.")