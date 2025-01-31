import streamlit as st
import pickle
import numpy as np

# Title of the app
st.title("Regression Model Inference App")

# Upload the pickle file
st.header("Upload the Model File")
uploaded_file = st.file_uploader("Upload a Linear Regression model (.pkl file)", type="pkl")

# Input fields for user to enter data
st.header("Enter Input Data")
input_data = []
for i in range(10):  # Adjust this if the dataset has a different number of features
    value = st.number_input(f"Feature {i + 1}", value=0.0, step=0.01)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    if uploaded_file is None:
        st.error("Please upload a valid model file.")
    else:
        try:
            # Load the model from the uploaded pickle file
            model = pickle.load(uploaded_file)

            # Convert input data to a 2D array
            input_data = np.array(input_data).reshape(1, -1)

            # Perform prediction
            prediction = model.predict(input_data)

            # Display the prediction result
            st.success(f"The predicted value is: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
