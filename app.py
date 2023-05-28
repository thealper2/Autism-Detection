import streamlit as st
import pickle
import numpy as np
import pandas as pd

model1 = pickle.load(open("logreg.pkl", "rb"))
model2 = pickle.load(open("svc.pkl", "rb"))
model3 = pickle.load(open("xgb.pkl", "rb"))

def find_asd(res):
    if res == 1:
        return "Austistic Spectrum Disorder Detected"
    else:
        return "Normal"

ethnicities = ["Asian", "Black", "Hispanic", "Latino", "Middle Eastern", "Others", "Pasifika", "South Asian", "Turkish", "White-European"]
relations = ["Health Care Professional", "Others", "Parent", "Relative", "Self"]
genders = ["Female", "Male"]


a1 = st.selectbox("A1 Score", [0, 1])
a2 = st.selectbox("A2 Score", [0, 1])
a3 = st.selectbox("A3 Score", [0, 1])
a4 = st.selectbox("A4 Score", [0, 1])
a5 = st.selectbox("A5 Score", [0, 1])
a6 = st.selectbox("A6 Score", [0, 1])
a7 = st.selectbox("A7 Score", [0, 1])
a8 = st.selectbox("A8 Score", [0, 1])
a9 = st.selectbox("A9 Score", [0, 1])
a10 = st.selectbox("A10 Score", [0, 1])
age = st.number_input("Age")
gender = st.selectbox("Gender", genders)
ethnicity = st.selectbox("Ethnicity", ethnicities)
jaundice = st.selectbox("Jaundice", [0, 1])
autism = st.selectbox("Autism", [0, 1])
used_app_before = st.selectbox("Used App Before", [0, 1])
result = st.number_input("Result")
relation = st.selectbox("Relation", relations)

if st.button("Detect"):
    gender = genders.index(gender)
    ethnicity = ethnicities.index(ethnicity)
    relation = relations.index(relation)
    test = np.array([[a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, age, gender, ethnicity, jaundice, autism, used_app_before, result, relation]])
    res1 = model1.predict(test)
    print(res1)
    res2 = model2.predict(test)
    print(res2)
    res3 = model3.predict(test)
    print(res3)
    result1 = find_asd(res1[0])
    result2 = find_asd(res2[0])
    result3 = find_asd(res3[0])
    st.success("Logistic Regression: " + result1)
    st.success("SVC: " + result2)
    st.success("XGBoost: " + result3)
