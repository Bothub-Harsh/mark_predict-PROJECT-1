import pickle
import numpy as np
import streamlit as st 

load_model=pickle.load(open("../models/model.pkl","rb"))

st.title("📊 Student Marks Predictor")
st.write("Enter study hours to predict marks")

enter = st.text_input("Enter Hours Studied")

if st.button("predict"):
    prediction = load_model.predict([[enter]])
    st.success(f"prediction marks: {prediction}")

# # Load model
# with open("../models/model.pkl","rb") as f:
#     loaded_model = pickle.load(f)

# enter = int(input("Enter your hour: "))
# pred = loaded_model.predict([[enter]])
# print(f"Prediction for {enter} hours:", pred)


