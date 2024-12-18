import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and preprocessor
with open('model4.pkl', 'rb') as file:
    model = pickle.load(file)

with open('preprosser_4.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# Inject custom CSS for advanced UI styling
st.markdown(
    """
    <style>
    /* Full-page gradient background */
    .stApp {
        background: linear-gradient(to right, #d7d2cc, #304352);
        font-family: 'Arial', sans-serif;
        color: white;
    }

    /* Title and header styling */
    h1, h2 {
        color: #ffcc00;
        text-align: center;
    }

    /* Center the main content */
    .main-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 20px;
    }

    /* Card design for input and result sections */
    .card {
        background-color: #ffffff;
        color: #333333;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        max-width: 500px;
        margin: 20px auto;
        text-align: center;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(to right, #ff9966, #ff5e62);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #ff5e62, #ff9966);
        transform: scale(1.1);
    }

    /* Footer styling */
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #dddddd;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title
st.title(" Customer Churn Prediction App")

# Subtitle
st.markdown("<div class='main-container'><h2> </h2></div>", unsafe_allow_html=True)

# Input section styled as a card
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Enter Customer Information")

gender = st.selectbox("Gender", ('Male', 'Female'))
InternetService = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
Contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=1)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0, max_value=200, value=50)
TotalCharges = st.number_input("Total Charges", min_value=0, max_value=10000, value=0)

st.markdown("</div>", unsafe_allow_html=True)

# Submit button
submitted = st.button("Predict")

# Prediction logic
if submitted:
    try:
        # Prepare input data
        custom_data_input_dict = {
            "gender": gender,
            "InternetService": InternetService,
            "Contract": Contract,
            "tenure": tenure,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        }
        data = pd.DataFrame(custom_data_input_dict, index=[0])

        # Transform input data
        transformed_data = preprocessor.transform(data)

        # Make prediction
        prediction = model.predict(transformed_data)
        prediction_proba = model.predict_proba(transformed_data)  # Get probabilities

        # Display results in a styled card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Prediction Result")
        if prediction[0] == 0:
            st.success("✅ This customer is likely to stay.")
        else:
            st.error("⚠️ This customer is likely to churn.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Plot probability scores
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Probability Scores")
        fig, ax = plt.subplots()
        classes = ['Stay', 'Churn']
        ax.bar(classes, prediction_proba[0], color=['#28a745', '#dc3545'])
        ax.set_ylabel("Probability")
        ax.set_title("Churn Prediction Probability")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown("""
<div class="footer">
     Customer Churn Prediction 
</div>
""", unsafe_allow_html=True)
