import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model
import pandas as pd
import pickle

#Load the trained model
model = tf.keras.models.load_model("model.h5")

#Load the trained, scaler pickle, ohe
model = load_model("model.h5")

#Load the encoder and scaler
with open("onehot_encoder.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    lable_encoder_gender = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


#Stream lit app
st.title("Customer Churn Prediction")

#User input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", lable_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 1, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_sr_card = st.selectbox("Has Credit Card", [0,1])
is_active_number = st.selectbox("Is Active Member", [0,1])

#Prepare Input data
input_data = pd.DataFrame({
    "CreditScore" : [credit_score], 
    'Gender' : [lable_encoder_gender.transform([gender])[0]], 

    "Age":[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_sr_card],
    'IsActiveMember':[is_active_number],
    'EstimatedSalary':[estimated_salary]
})

#One hot encode "Geopraphy"
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))


#Combining ohe with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#Scaling the Input data
input_scaled = scaler.transform(input_data)

#Predict Churn
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

st.write(f"Churn Probability : {prediction_prob:.2f}")
if prediction_prob > 0.5 :
    st.write("The custmer is likely to churn.")
else:
    st.write("The custmer is not likely to churn.")