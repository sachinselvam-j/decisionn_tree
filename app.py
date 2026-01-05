import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("decision_tree_car_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚗 Car Evaluation – Decision Tree")

# Encoding dictionaries (VERY IMPORTANT)
buying_map = {"vhigh": 3, "high": 2, "med": 1, "low": 0}
maint_map = {"vhigh": 3, "high": 2, "med": 1, "low": 0}
doors_map = {"2": 0, "3": 1, "4": 2, "5more": 3}
persons_map = {"2": 0, "4": 1, "more": 2}
lug_map = {"small": 0, "med": 1, "big": 2}
safety_map = {"low": 0, "med": 1, "high": 2}

# User inputs
buying = st.selectbox("Buying Price", list(buying_map.keys()))
maint = st.selectbox("Maintenance Price", list(maint_map.keys()))
doors = st.selectbox("Number of Doors", list(doors_map.keys()))
persons = st.selectbox("Persons Capacity", list(persons_map.keys()))
lug_boot = st.selectbox("Luggage Boot", list(lug_map.keys()))
safety = st.selectbox("Safety Level", list(safety_map.keys()))

if st.button("Predict"):
    input_df = pd.DataFrame([[
        buying_map[buying],
        maint_map[maint],
        doors_map[doors],
        persons_map[persons],
        lug_map[lug_boot],
        safety_map[safety]
    ]], columns=["buying", "maint", "doors", "persons", "lug_boot", "safety"])

    prediction = model.predict(input_df)

    st.success(f"Prediction Result: **{prediction[0]}**")
