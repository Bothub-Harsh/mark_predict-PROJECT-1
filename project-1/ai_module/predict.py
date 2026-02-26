import pickle
import numpy as np
import os
import streamlit as st 

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

with open(model_path, "rb") as f:
    load_model = pickle.load(f)

st.title("📊 Student Marks Predictor")
st.write("Enter study hours to predict marks")

enter = st.number_input("Enter Hours Studied", min_value=0.0)

if st.button("Predict"):
    prediction = load_model.predict(np.array([[enter]]))
    st.success(f"Prediction Marks: {prediction[0]:.2f}")

# # Load model
# with open("../models/model.pkl","rb") as f:
#     loaded_model = pickle.load(f)

# enter = int(input("Enter your hour: "))
# pred = loaded_model.predict([[enter]])
# print(f"Prediction for {enter} hours:", pred)


