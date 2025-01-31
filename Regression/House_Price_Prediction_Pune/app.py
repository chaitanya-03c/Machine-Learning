import streamlit as st
import pickle
import numpy as np

# Load the model and columns
with open('columns.pkl', 'rb') as columns_file:
    data_columns = pickle.load(columns_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

locations = data_columns['locations']
area_type = data_columns['area_types']
availability = data_columns['availabilities']

# Extract options for dropdowns

if not locations or not area_type or not availability:
    st.error("Dropdown options are not properly loaded. Please check the 'columns.pkl' file structure.")

# Define prediction function
def prediction(location, bhk, bath, balcony, sqft, area_type, availability):
    x = np.zeros(len(data_columns['data_columns']))
    x[0] = bath
    x[1] = balcony
    x[2] = bhk
    x[3] = sqft

    # Handle location
    if location and f'location_{location.lower()}' in data_columns['data_columns']:
        loc_index = data_columns['data_columns'].index(f'location_{location.lower()}')
        x[loc_index] = 1

    # Handle area type
    if area_type and f'area_type_{area_type}' in data_columns['data_columns']:
        area_index = data_columns['data_columns'].index(f'area_type_{area_type}')
        x[area_index] = 1

    # Handle availability
    if availability and f'availability_{availability}' in data_columns['data_columns']:
        avail_index = data_columns['data_columns'].index(f'availability_{availability}')
        x[avail_index] = 1

    return round(model.predict([x])[0], 2)

# Streamlit App UI
st.title("Pune House Price Prediction")

st.sidebar.header("Input Features")

# Dropdown for Location
location = st.sidebar.selectbox("Location", ["Select"] + sorted(locations))

# Dropdown for Area Type
area_type = st.sidebar.selectbox("Area Type", ["Select"] + sorted(area_type))

# Dropdown for Availability
availability = st.sidebar.selectbox("Availability", ["Select"] + sorted(availability))

# Number inputs
bhk = st.sidebar.number_input("Number of Bedrooms (BHK)", min_value=1, step=1)
bath = st.sidebar.number_input("Number of Bathrooms", min_value=1, step=1)
balcony = st.sidebar.number_input("Number of Balconies", min_value=0, step=1)
sqft = st.sidebar.number_input("Total Square Feet", min_value=300, step=10)

# Predict button
if st.sidebar.button("Predict Price"):
    if location == "Select" or area_type == "Select" or availability == "Select":
        st.error("Please select valid options for location, area type, and availability.")
    else:
        price = prediction(location, bhk, bath, balcony, sqft, area_type, availability)
        st.write(f"### The predicted price is ₹{price} Lacs")

st.write("---")
st.write("This app predicts house prices based on location, size, and other features.")
