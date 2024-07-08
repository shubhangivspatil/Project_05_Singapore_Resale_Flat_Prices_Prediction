import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Define file paths
model_path = 'D:/GUVI_Projects/My_Projects/singapur sheets/decision_tree_model.joblib'
data_path = 'D:/GUVI_Projects/My_Projects/singapur sheets/cleaned_singapore_resale_flat_prices.csv'
columns_path = 'D:/GUVI_Projects/My_Projects/singapur sheets/columns.pkl'
towns_path = 'D:/GUVI_Projects/My_Projects/singapur sheets/towns.csv'
flat_types_path = 'D:/GUVI_Projects/My_Projects/singapur sheets/flat_types.csv'
storey_ranges_path = 'D:/GUVI_Projects/My_Projects/singapur sheets/storey_ranges.csv'
flat_models_path = 'D:/GUVI_Projects/My_Projects/singapur sheets/flat_models.csv'
street_names_path = 'D:/GUVI_Projects/My_Projects/singapur sheets/street_names.csv'

# Function to save dropdown options if they don't exist
def save_dropdown_options(data):
    towns = sorted(data['town'].unique())
    flat_types = sorted(data['flat_type'].unique())
    storey_ranges = sorted(data['storey_range'].unique())
    flat_models = sorted(data['flat_model'].unique())
    street_names = sorted(data['street_name'].unique())

    pd.Series(towns).to_csv(towns_path, index=False)
    pd.Series(flat_types).to_csv(flat_types_path, index=False)
    pd.Series(storey_ranges).to_csv(storey_ranges_path, index=False)
    pd.Series(flat_models).to_csv(flat_models_path, index=False)
    pd.Series(street_names).to_csv(street_names_path, index=False)

    print("Dropdown options saved successfully.")

# Function to train and save the model and column names if they don't exist
def train_and_save_model():
    try:
        # Load a sample of the dataset to reduce memory usage
        data = pd.read_csv(data_path, nrows=5000)  # Adjust nrows as necessary for your system

        # Convert data types to reduce memory usage
        data['floor_area_sqm'] = data['floor_area_sqm'].astype('float32')
        data['lease_commence_date'] = data['lease_commence_date'].astype('int16')
        data['resale_price'] = data['resale_price'].astype('float32')

        # Prepare the features (X) and target (y)
        X = data[['floor_area_sqm', 'storey_range', 'lease_commence_date', 'year', 'month', 'flat_model', 'town', 'flat_type', 'street_name']]
        y = data['resale_price']

        # One-hot encoding for categorical features
        X = pd.get_dummies(X, columns=['storey_range', 'flat_model', 'town', 'flat_type', 'street_name'], drop_first=True)

        # Save the column names
        joblib.dump(X.columns, columns_path)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the Decision Tree model
        decision_tree_model = DecisionTreeRegressor(random_state=42)

        # Train the model
        decision_tree_model.fit(X_train, y_train)

        # Save the model to disk using Joblib
        joblib.dump(decision_tree_model, model_path)

        print(f"Model and column names saved successfully at {model_path} and {columns_path}.")
    except MemoryError:
        print("MemoryError: Could not allocate memory for the operation. Try reducing the dataset size.")

# Check and train model if not exist
if not os.path.exists(model_path) or not os.path.exists(columns_path):
    train_and_save_model()

# Load the trained model and columns
model = joblib.load(model_path)
columns = joblib.load(columns_path)

# Check and save dropdown options if not exist
if not (os.path.exists(towns_path) and os.path.exists(flat_types_path) and os.path.exists(storey_ranges_path) and os.path.exists(flat_models_path) and os.path.exists(street_names_path)):
    data = pd.read_csv(data_path, nrows=5000)  # Adjust nrows as necessary for your system
    save_dropdown_options(data)

# Load dropdown options
towns = pd.read_csv(towns_path).squeeze().tolist()
flat_types = pd.read_csv(flat_types_path).squeeze().tolist()
storey_ranges = pd.read_csv(storey_ranges_path).squeeze().tolist()
flat_models = pd.read_csv(flat_models_path).squeeze().tolist()

# Streamlit app
st.title("Singapore Resale Flat Prices Predicting")
st.markdown("""
### Skills Takeaway from This Project
- Data Wrangling
- EDA
- Model Building
- Model Deployment

### Domain
- Real Estate

### Project Objective
The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

### How to use this application:
1. Select the town, flat type, storey range, flat model, and street name from the dropdown menus.
2. Adjust the sliders for floor area (sqm), lease commence date, month, and year.
3. View the predicted resale price based on the inputs.
""")

# Define the input features for the model
def user_input_features():
    town = st.selectbox('Town', towns)
    flat_type = st.selectbox('Flat Type', flat_types)
    storey_range = st.selectbox('Storey Range', storey_ranges)
    floor_area_sqm = st.slider('Floor Area (sqm)', 20, 200, 50)
    flat_model = st.selectbox('Flat Model', flat_models)
    lease_commence_date = st.slider('Lease Commence Date', 1960, 2022, 1990)
    month = st.slider('Month', 1, 12, 1)
    year = st.slider('Year', 1990, 2024, 2024)
    
    data = {
        'town': town,
        'flat_type': flat_type,
        'storey_range': storey_range,
        'floor_area_sqm': floor_area_sqm,
        'flat_model': flat_model,
        'lease_commence_date': lease_commence_date,
        'month': month,
        'year': year
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Encode input features and align with the model's training columns
input_df_encoded = pd.get_dummies(input_df)
input_df_encoded = input_df_encoded.reindex(columns=columns, fill_value=0)

# Debug: Print the encoded input dataframe
st.write("Encoded Input DataFrame:")
st.write(input_df_encoded)

# Predict resale price
try:
    prediction = model.predict(input_df_encoded)
    formatted_prediction = f"${prediction[0]:,.2f}"
    st.subheader('Predicted Resale Price')
    st.write(formatted_prediction)
except ValueError as e:
    st.error(f"Error in prediction: {e}")

# Display model info
st.sidebar.header("Model Info")
st.sidebar.write(f"Model: DecisionTreeRegressor")
st.sidebar.write(f"Training Data Size: 5000 rows")
st.sidebar.write(f"Random State: 42")

# Display additional debug info if necessary
if st.sidebar.checkbox("Show Encoded Input DataFrame"):
    st.write("Encoded Input DataFrame:")
    st.write(input_df_encoded)
