import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Asteroid Diameter Predictor",
    page_icon="☄️",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("☄️ Asteroid Diameter Predictor")

st.write(
    "Enter the asteroid characteristics below to predict "
    "its diameter using a trained Deep Neural Network."
)


# --------------------------------------------------
# LOAD MODEL AND PREPROCESSING FILES
# --------------------------------------------------

@st.cache_resource
def load_prediction_objects():

    model = load_model("asteroid_dnn_model.keras")

    with open("x_imputer.pkl", "rb") as f:
        imputer = pickle.load(f)

    with open("x_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    return model, imputer, scaler, y_scaler


model, imputer, scaler, y_scaler = load_prediction_objects()


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.subheader("Enter Asteroid Information")


H = st.number_input(
    "Absolute Magnitude (H)",
    min_value=0.0,
    max_value=40.0,
    value=15.0
)

a = st.number_input(
    "Semi-major Axis (a)",
    min_value=0.0,
    value=2.5
)

e = st.number_input(
    "Eccentricity (e)",
    min_value=0.0,
    max_value=1.0,
    value=0.1
)

i = st.number_input(
    "Inclination (i)",
    min_value=0.0,
    value=5.0
)

q = st.number_input(
    "Perihelion Distance (q)",
    min_value=0.0,
    value=2.0
)


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button("🔮 Predict Diameter"):

    # Create dataframe
    input_data = pd.DataFrame({
        "H": [H],
        "a": [a],
        "e": [e],
        "i": [i],
        "q": [q]
    })

    # Apply preprocessing
    input_imputed = imputer.transform(input_data)

    input_scaled = scaler.transform(input_imputed)

    # Prediction
    prediction_scaled = model.predict(input_scaled)

    # Convert prediction back to original scale
    prediction = y_scaler.inverse_transform(
        prediction_scaled
    )

    predicted_diameter = prediction[0][0]

    # Display result
    st.success(
        f"Predicted Asteroid Diameter: "
        f"{predicted_diameter:.2f}"
    )

    st.info(
        "The prediction is generated using the trained "
        "Deep Neural Network regression model."
    )