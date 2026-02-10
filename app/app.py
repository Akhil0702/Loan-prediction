import pandas as pd
import pickle
import streamlit as st
import numpy as np

# -----------------------
# Load saved objects
# -----------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open("le_target.pkl", "rb") as f:
    le_target = pickle.load(f)

st.title("Loan Prediction App")

# -----------------------
# User Input Form
# -----------------------
def user_input_form():
    with st.form("loan_form"):
        st.subheader("Enter Applicant Details")

        data = {}
        data['Gender'] = st.selectbox("Gender", ['Male', 'Female'])
        data['Married'] = st.selectbox("Married", ['Yes', 'No'])
        data['Dependents'] = st.selectbox("Dependents", ['0','1','2','3+'])
        data['Education'] = st.selectbox("Education", ['Graduate', 'Not Graduate'])
        data['Self_Employed'] = st.selectbox("Self Employed", ['Yes', 'No'])

        # ------------ Income fields (text input, no arrows) ------------
        applicant_income = st.text_input("Applicant Income (per month) - numbers only", "5000")
        coapplicant_income = st.text_input("Coapplicant Income (per month) - numbers only", "0")

        # Convert text input safely
        data['ApplicantIncome'] = int(applicant_income) if applicant_income.isdigit() else 0
        data['CoapplicantIncome'] = int(coapplicant_income) if coapplicant_income.isdigit() else 0

        # ------------ Loan Amount Slider ------------
        data['LoanAmount'] = st.slider(
            "Loan Amount ($)",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=500
        )

        data['Loan_Amount_Term'] = st.number_input("Loan Amount Term (months)", 0, 480, 360)
        data['Credit_History'] = st.selectbox("Credit History", [1.0, 0.0])
        data['Property_Area'] = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

        submit = st.form_submit_button("Submit")

    return pd.DataFrame([data]), submit


input_df, submitted = user_input_form()

# -----------------------
# Run prediction only after Submit
# -----------------------
if submitted:
    train_columns = list(model.feature_names_in_)

    for col in train_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[train_columns]

    for col in label_encoders:
        if col in input_df.columns:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col].astype(str))

    num_cols = input_df.select_dtypes(include=np.number).columns
    input_df[num_cols] = input_df[num_cols].fillna(0)

    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    st.success("Loan Approved ✅" if prediction[0] == 1 else "Loan Not Approved ❌")

    st.subheader("Prediction Probability")
    st.write(prediction_proba)
