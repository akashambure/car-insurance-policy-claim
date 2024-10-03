import numpy as np  
import pandas as pd  
import joblib  
import streamlit as st 
import sklearn 

# Set up the page  
st.set_page_config(page_title='Car Insurance Claim Prediction System', layout='centered')  

# Load the model  
@st.cache_resource  
def load_model():  
    model = joblib.load('pipe.pkl')  
    return model  

model = load_model()  

# Title & Description  
st.title('🚗 Car Insurance Claim Prediction')  
st.write("""  
This application predicts whether a customer will claim car insurance within the next three months.  
Please provide the following information:  
""")  

# Define assumed min and max values used during normalization  
policy_tenure_min = 0.0   # in years  
policy_tenure_max = 10.0  # in years  

age_of_car_min = 0.0      # in years  
age_of_car_max = 20.0     # in years  

age_of_policyholder_min = 18.0  # in years  
age_of_policyholder_max = 85.0  # in years  

# Function to normalize input features  
def normalize_value(value, min_value, max_value):  
    return (value - min_value) / (max_value - min_value)  

# User input features  
def get_user_input():  
    # Collect user inputs in years as float  
    policy_tenure_years = st.number_input('Policy Tenure (in years)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)  
    age_of_car_years = st.number_input('Age of Car (in years)', min_value=0.0, max_value=20.0, value=5.0, step=0.1)  
    age_of_policyholder_years = st.number_input('Age of Policyholder (in years)', min_value=18.0, max_value=85.0, value=35.0, step=0.1)  
    area_cluster = st.selectbox('Area Cluster', options=[  
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C11', 'C12',  
        'C13', 'C14', 'C15', 'C16', 'C18', 'C19', 'C20', 'C21', 'C22'  
    ], index=0)  
    population_density = st.number_input('Population Density', min_value=50.0, max_value=27003.0, value=500.0)  
    
    # Normalize inputs  
    policy_tenure_normalized = normalize_value(policy_tenure_years, policy_tenure_min, policy_tenure_max)  
    age_of_car_normalized = normalize_value(age_of_car_years, age_of_car_min, age_of_car_max)  
    age_of_policyholder_normalized = normalize_value(age_of_policyholder_years, age_of_policyholder_min, age_of_policyholder_max)  
    
    # Process area_cluster to remove 'C' and convert to integer  
    area_cluster_number = int(area_cluster.replace('C', ''))  
    
    user_data = {  
        'policy_tenure': policy_tenure_normalized,  
        'age_of_car': age_of_car_normalized,  
        'age_of_policyholder': age_of_policyholder_normalized,  
        'area_cluster': area_cluster_number,  
        'population_density': population_density  
    }  
    
    features = pd.DataFrame(user_data, index=[0])  

    # Ensure correct data types  
    features['policy_tenure'] = features['policy_tenure'].astype(float)  
    features['age_of_car'] = features['age_of_car'].astype(float)  
    features['age_of_policyholder'] = features['age_of_policyholder'].astype(float)  
    features['population_density'] = features['population_density'].astype(float)  
    features['area_cluster'] = features['area_cluster'].astype(int)  # Now an integer  
    
    return features  

user_input_df = get_user_input()  


# Add a checkbox to display user input  
if st.checkbox('Show User Input'):  
    st.subheader('User Input:')  
    st.write(user_input_df)  

# Add a button to trigger prediction  

st.subheader('Will the customer claim policy within 3 months ?') 
 
if st.button('Predict'):  
    # Make prediction  
    prediction = model.predict(user_input_df)  
    
    # Map prediction to 'Yes' or 'No'  
    prediction_result = 'Yes' if prediction[0] == 1 else 'No'  
    
    # Display prediction with color  
    if prediction_result == 'Yes':  
        st.markdown(f"<h1 style='color: red;'> {prediction_result}</h1>", unsafe_allow_html=True)  
    else:  
        st.markdown(f"<h1 style='color: green;'>{prediction_result}</h1>", unsafe_allow_html=True)  
