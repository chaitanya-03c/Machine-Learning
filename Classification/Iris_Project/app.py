import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
model_path = 'log_model.pkl'
scaler_path = 'scaler.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Mapping target to Iris species
target_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# Streamlit App
st.title("Iris Species Predictor")
st.write("Enter the values for the features below to predict the Iris species.")

# Input fields for features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
    scaled_input = scaler.transform(input_features)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    species = target_mapping.get(prediction, "Unknown")
    
    # Display result
    st.success(f"The predicted Iris species is: **{species}**")
