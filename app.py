import streamlit as st
import numpy as np
import joblib
import pandas as pd

force_model = joblib.load("models/force_model.pkl")
orient_model = joblib.load("models/orientation_model.pkl")
curv_model = joblib.load("models/curvature_model.pkl")

st.title("Multi-Modal Sensor State Prediction")

st.write("Input Sensor Signals")

signals = []
for i in range(1,6):
    val = st.slider(f"S{i}", -2.0, 5.0, 0.0)
    signals.append(val)

# signals_array = np.array(signals).reshape(1,-1)
signals_df = pd.DataFrame(
    [signals],
    columns=["S1","S2","S3","S4","S5"]
)
# Step 1: Predict Force
pred_force = force_model.predict(signals_df)


# Step 2: Cascaded Input
cascade_input = signals_df.copy()
cascade_input["PredictedForce"] = pred_force

# Step 3: Predict Orientation & Curvature
pred_orient = orient_model.predict(cascade_input)
pred_curv = curv_model.predict(cascade_input)

st.subheader("Predicted Physical State")
st.write("Force (N):", round(pred_force[0], 3))
st.write("Curvature:", round(pred_curv[0], 3))
st.write("Orientation (degrees):", pred_orient[0])