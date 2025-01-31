import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
model_path = 'model.pickle'
scaler_path = 'scaler.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit App
st.title("Titanic Survival Predictor")
st.write("Enter the details below to predict if the passenger survived or not.")

# Input fields for features
passenger_id = st.number_input("Passenger ID", min_value=1, value=1, step=1)
pclass = st.selectbox("Pclass (Ticket Class)", options=[1, 2, 3], format_func=lambda x: f"Class {x}")
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0, step=1)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0, step=1)
fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
embarked = st.selectbox("Port of Embarkation", options=[0, 1, 2], format_func=lambda x: ["Cherbourg", "Queenstown", "Southampton"][x])

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_features = np.array([passenger_id, pclass, sex, age, sibsp, parch, fare, embarked]).reshape(1, -1)
    scaled_input = scaler.transform(input_features)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    result = "Survived" if prediction == 1 else "Not Survived"
    
    # Display result
    st.success(f"The passenger is predicted to have: **{result}**")
