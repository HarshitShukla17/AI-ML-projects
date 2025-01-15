import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Load the model
model = tf.keras.models.load_model("model.h5")

# App title and description
st.title("Customer Churn Prediction")
st.markdown("""
This app predicts whether a customer is likely to churn based on their profile information.  
You can either input individual data or upload a file to predict churn for multiple customers and visualize the data.
""")

# File upload widget
uploaded_file = st.file_uploader("Upload your customer data (CSV file)", type=["csv", "xlsx"])

# If a file is uploaded
if uploaded_file is not None:
    # Load the dataset
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # Display the first few rows of the dataset
    #st.write("### Data Preview", data.head())

    # Show basic statistics of the dataset
    #st.write("### Dataset Summary", data.describe())

    # Convert categorical columns to numerical
    index=data.iloc[:,1]  # Save the CustomerId column
    data = data.iloc[:,3:]  # Remove the CustomerId column
    categorical_columns = ['Geography', 'Gender']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Ensure the dataset has the required columns for prediction
    required_columns = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "Geography_Germany", "Geography_Spain", "Gender_Male"
    ]
    for col in required_columns:
        if col not in data.columns:
            data[col] = 0  # Add missing columns with default value 0

    # Reorder the columns
    data = data[required_columns]

    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Make predictions for all rows
    predictions = model.predict(data_scaled)
    data['Churn_Probability'] = predictions
    result_data = pd.DataFrame({
        'CustomerId': index,
        'Churn_Probability': data['Churn_Probability']
    })

    # Display predictions in a table
    st.write("### Predictions Table", result_data)

    
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable warning
    # Show a pie chart of churn vs. non-churn
    st.write("### Churn vs. Non-Churn Pie Chart")
    churn_labels = ['Churn', 'No Churn']
    churn_counts = data['Churn_Probability'].apply(lambda x: 'Churn' if x >= 0.5 else 'No Churn').value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(churn_counts, labels=churn_labels, autopct='%1.1f%%', colors=['red', 'green'])
    st.pyplot()

    # Show the distribution of churn probabilities
    st.write("### Churn Probability Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Churn_Probability'], kde=True, color="blue")
    st.pyplot()

    # Display a bar chart showing churn vs. non-churn counts
    st.write("### Churn vs No Churn Counts (Bar Chart)")
    churn_count = data['Churn_Probability'].apply(lambda x: 'Churn' if x >= 0.5 else 'No Churn').value_counts()
    churn_count.plot(kind='bar', color=["red", "green"])
    st.pyplot()

# Else, if user chooses to input a single data point
else:
    with st.form("churn_form"):
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
        # Convert categorical inputs to numerical values
        geography_dict = {"France": 0, "Spain": 1, "Germany": 2}
        gender_dict = {"Male": 0, "Female": 1}

        geography_num = geography_dict[geography]
        gender_num = gender_dict[gender]

        # Create a DataFrame for the input
        input_df = pd.DataFrame({
            "CreditScore": [credit_score],
            "Geography": [geography_num],
            "Gender": [gender_num],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        })

        # Use get_dummies to encode categorical variables
        input_df = pd.get_dummies(input_df, columns=["Geography", "Gender"], drop_first=True)

        # Ensure the input data has the same columns as the training data
        required_columns = [
            "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary",
            "Geography_Germany", "Geography_Spain", "Gender_Male"
        ]
        for col in required_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns with default value 0

        # Reorder the columns to match the model's input
        input_df = input_df[required_columns]

        # Scale the input data
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_df)

        # Predict churn
        prediction = model.predict(input_data_scaled)[0]
        probability = prediction[0]  # Assuming the model output is a probability

        # Display the result with styling
        st.write("### Prediction Results")
        if probability >= 0.5:
            st.error(f"The customer is likely to churn. Probability: {probability:.2f}")
        else:
            st.success(f"The customer is not likely to churn. Probability: {probability:.2f}")
