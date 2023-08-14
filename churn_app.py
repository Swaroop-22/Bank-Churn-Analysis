import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from numpy import where

model = XGBClassifier()
model.load_model('xgb_model.h5')  

# Define a function to predict churn
def predict_churn(customer):
    churn_probability = model.predict_proba(customer)[0][1]
    churn_prediction = model.predict(customer)[0]
    return churn_prediction, churn_probability

st.title('Churn Prediction')
st.write('Enter customer details to predict churn')

credit_score = st.number_input('Credit Score')
country = st.selectbox('Country', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 1, 100)
tenure = st.number_input('Tenure')
balance = st.number_input('Balance')
products_number = st.number_input('Number of Products')
credit_card = st.selectbox('Has Credit Card 1=(Yes) 0=(No)', [1, 0])
active_member = st.selectbox('Is Active Member 1=(Yes) 0=(No)', [1, 0])
estimated_salary = st.number_input('Estimated Salary $ per annum ')

# Create a dictionary of customer data
customer_data = {
    'CreditScore': credit_score,
    'Country': country,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': products_number,
    'HasCrCard': credit_card,
    'IsActiveMember': active_member,
    'EstimatedSalary': estimated_salary
}

# Convert the dictionary to a DataFrame
customer_df = pd.DataFrame([customer_data])

# Label encode categorical variables
labelencoder = LabelEncoder()
customer_df['Country'] = labelencoder.fit_transform(customer_df['Country'])
customer_df['Gender'] = labelencoder.fit_transform(customer_df['Gender'])

# Scale numerical variables
scaler = MinMaxScaler(feature_range=(0,1))
customer_df = pd.DataFrame(scaler.fit_transform(customer_df), columns=customer_df.columns)

if st.button('Predict Churn'):
    churn, probability = predict_churn(customer_df.values)
    if churn:
        st.write('This customer is likely to churn with a probability of ')
    else:
        st.write('This customer is not likely to churn with a probability of ')



#   streamlit run "C:\Users\Hi\Desktop\DS_Projects\DS_WORK_PROJECTS\Bank_Churn_Analysis\churn_app.py"