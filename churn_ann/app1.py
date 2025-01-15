import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

# Load the model
model = tf.keras.models.load_model("model.h5")

# App title and description
st.title("Customer Churn Prediction")
st.markdown("""
This app predicts whether a customer is likely to churn based on their profile information.  
You can either input individual data or upload a file to predict churn for multiple customers and visualize the data.
""")

# File upload widget
uploaded_file = st.file_uploader("Upload your customer data (CSV or Excel file)", type=["csv", "xlsx"])

# Process file input
if uploaded_file is not None:
    # Load the dataset
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Extract CustomerId and preprocess data
    index = data.iloc[:, 1]
    data = data.iloc[:, 3:]
    categorical_columns = ["Geography", "Gender"]
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Ensure all required columns are present
    required_columns = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "Geography_Germany", "Geography_Spain", "Gender_Male"
    ]
    for col in required_columns:
        if col not in data.columns:
            data[col] = 0

    # Reorder columns and scale data
    data = data[required_columns]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Predict churn probabilities
    predictions = model.predict(data_scaled)
    data["Churn_Probability"] = predictions
    result_data = pd.DataFrame({
        "CustomerId": index,
        "Churn_Probability": data["Churn_Probability"]
    })

    # Interactive Table using AgGrid
    st.subheader("Predictions Table")
    grid_builder = GridOptionsBuilder.from_dataframe(result_data)
    grid_builder.configure_pagination(enabled=True)
    grid_builder.configure_side_bar()
    grid_options = grid_builder.build()
    AgGrid(result_data, gridOptions=grid_options, theme="streamlit")

    # Pie chart visualization
    st.subheader("Churn vs Non-Churn Distribution")
    churn_labels = ["Churn", "No Churn"]
    churn_counts = data["Churn_Probability"].apply(lambda x: "Churn" if x >= 0.5 else "No Churn").value_counts()
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.pie(churn_counts, labels=churn_labels, autopct="%1.1f%%", colors=["red", "green"], startangle=90)
    st.pyplot(fig1)

    # Churn Probability Distribution
    st.subheader("Churn Probability Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(data["Churn_Probability"], kde=True, color="blue", ax=ax2)
    ax2.set_title("Distribution of Churn Probabilities")
    st.pyplot(fig2)

    # Bar chart for churn counts
    st.subheader("Churn Counts (Bar Chart)")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    churn_counts.plot(kind="bar", color=["red", "green"], ax=ax3)
    ax3.set_ylabel("Number of Customers")
    ax3.set_title("Churn vs No Churn Counts")
    st.pyplot(fig3)

else:
    st.subheader("Single Customer Prediction")
    with st.form("churn_form"):
        # Input fields
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=5)
        balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
        num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
        has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
        is_active_member = st.selectbox("Is Active Member?", [0, 1])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")

        # Submit button
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input for the model
        geography_dict = {"France": 0, "Spain": 1, "Germany": 2}
        gender_dict = {"Male": 0, "Female": 1}
        input_df = pd.DataFrame({
            "CreditScore": [credit_score],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
            "Geography_Germany": [1 if geography == "Germany" else 0],
            "Geography_Spain": [1 if geography == "Spain" else 0],
            "Gender_Male": [1 if gender == "Male" else 0]
        })

        # Scale data
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_df)

        # Predict churn probability
        probability = model.predict(input_data_scaled)[0][0]

        # Display result
        st.subheader("Prediction Results")
        if probability >= 0.5:
            st.error(f"The customer is likely to churn. Probability: {probability:.2f}")
        else:
            st.success(f"The customer is not likely to churn. Probability: {probability:.2f}")
