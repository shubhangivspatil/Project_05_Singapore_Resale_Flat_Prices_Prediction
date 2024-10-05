import streamlit as st
import pandas as pd
import joblib
from sklearn.exceptions import InconsistentVersionWarning

# Paths to model and dataset
model_path = 'D:/Final_Projects/project_new/random_forest_model.joblib'
data_path = 'D:/Final_Projects/project_new/latest_cleaned_singapore_resale_flat_prices.csv'

# Function to load model using memory mapping to avoid memory issues
def load_model_with_memory_mapping(model_path):
    try:
        # Use memory-mapping mode to avoid large memory allocations
        model = joblib.load(model_path, mmap_mode='r')  
        st.success("Model loaded successfully.")
        return model
    except InconsistentVersionWarning as w:
        original_version = w.original_sklearn_version
        st.error(f"Model was trained with scikit-learn version: {original_version}.")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load the model
model = load_model_with_memory_mapping(model_path)

# Load the cleaned dataset
try:
    data = pd.read_csv(data_path)
    st.success("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")

# Debug: Show the first few rows of the dataset to verify its structure
st.write("Dataset Preview:", data.head())

# Introduction section with project details
st.title("Singapore Resale Flat Prices Predicting")
st.markdown(""" 
**Skills Takeaway From This Project:**
- Data Wrangling
- Exploratory Data Analysis (EDA)
- Model Building
- Model Deployment

**Domain:**
- Real Estate

**Problem Statement:**
- The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

**Creator:**
- Shubhangi Patil

**Project:**
- Data Science

**GitHub Link:**
- [GitHub Repository](https://github.com/shubhangivspatil)
""")

# Get unique values for categorical features from the cleaned dataset
storey_range_options = data['storey_range'].unique().tolist()
town_options = data['town'].unique().tolist()
flat_type_options = data['flat_type'].unique().tolist()
flat_model_options = data['flat_model'].unique().tolist()
street_name_options = data['street_name'].unique().tolist()
year_options = data['year'].unique().tolist()  # Get unique years

# Debug: Show unique year options
st.write("Unique Year Options:", year_options)

# Streamlit app title
st.title("Singapore Resale Flat Price Prediction")

# User input form
with st.form("flat_details"):
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=10.0, max_value=200.0, step=0.1)
    
    storey_range = st.selectbox("Storey Range", options=storey_range_options)
    
    lease_commence_date = st.number_input("Lease Commence Year", min_value=1900, max_value=2100, value=1977, step=1)
    
    town = st.selectbox("Town", options=town_options)
    
    flat_type = st.selectbox("Flat Type", options=flat_type_options)
    
    flat_model = st.selectbox("Flat Model", options=flat_model_options)
    
    street_name = st.selectbox("Street Name", options=street_name_options)

    # Add year input; user can select from available years in the dataset
    year = st.selectbox("Year of Sale", options=year_options)  
    
    submit = st.form_submit_button("Predict Resale Price")

if submit:
    # Create the input data DataFrame including year and excluding month
    input_data = pd.DataFrame({
        'floor_area_sqm': [floor_area_sqm],               # Area of the flat in square meters
        'storey_range': [storey_range],                   # Storey range selected by user
        'lease_commence_date': [lease_commence_date],     # Lease commencement year
        'year': [year],                                   # Year of sale provided by user
        'flat_model': [flat_model],                       # Flat model selected by user
        'town': [town],                                   # Town selected by user
        'flat_type': [flat_type],                         # Flat type selected by user
        'street_name': [street_name]                      # Street name selected by user
    })

    # Debug: Show input data to verify correctness
    st.write("Input Data:", input_data)

    # Encode the input data using one-hot encoding for categorical features
    input_data_encoded = pd.get_dummies(input_data, columns=['storey_range', 'flat_model', 'town', 'flat_type', 'street_name'])

    # Debug: Show encoded input data to verify correctness
    st.write("Encoded Input Data:", input_data_encoded)

    # Ensure all expected columns are present by adding missing columns with zeros
    expected_columns = model.feature_names_in_  # Use feature names from the trained model

    # Debug: Show expected columns for prediction
    st.write("Expected Columns:", expected_columns)

    # Add missing columns with zeros if they are not present in the encoded DataFrame
    for col in expected_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

    # Reorder columns to match expected order from the trained model
    input_data_encoded = input_data_encoded[expected_columns]

    # Debug: Show final encoded input data before prediction
    st.write("Final Encoded Input Data for Prediction:", input_data_encoded)

    # Make the prediction using the trained model
    try:
        predicted_price = model.predict(input_data_encoded)[0]
        st.success(f"The predicted resale price for the flat is: SGD {predicted_price:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
