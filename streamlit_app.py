import streamlit as st
from joblib import load  # Or import pickle if you used that
import numpy as np

# Load the saved model
model = load('model.joblib')

# Define a function for prediction
def predict(input_data):
    # Assuming input_data is preprocessed and ready for prediction
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
st.title('My Machine Learning Model App')

# Assuming you're collecting user input for prediction
input_value = st.number_input('Enter a value for prediction')

# Button to trigger the prediction
if st.button('Predict'):
    # Make sure the input value is in the correct format for your model
    input_data = np.array([[input_value]])  # Example for single input
    prediction = predict(input_data)
    st.write(f'The prediction is: {prediction[0]}')
