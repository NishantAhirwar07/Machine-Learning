import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title='Weather Forecasting App', page_icon='🌦️', layout='centered')

st.title('🌦️ Weather Forecasting Model')
st.write('Predict whether rainfall will occur or not using trained ML model.')

MODEL_FILE = 'rainfall_prediction_model.pkl'

if not os.path.exists(MODEL_FILE):
    st.error('Model file not found! Please keep rainfall_prediction_model.pkl in same folder.')
    st.stop()

with open(MODEL_FILE, 'rb') as file:
    model_data = pickle.load(file)

model = model_data['model']
feature_names = model_data['feature_names']

st.sidebar.header('Enter Weather Details')
pressure = st.sidebar.number_input('Pressure', value=1015.9)
dewpoint = st.sidebar.number_input('Dew Point', value=19.9)
humidity = st.sidebar.number_input('Humidity', value=95.0)
cloud = st.sidebar.number_input('Cloud', value=81.0)
sunshine = st.sidebar.number_input('Sunshine', value=0.0)
winddirection = st.sidebar.number_input('Wind Direction', value=40.0)
windspeed = st.sidebar.number_input('Wind Speed', value=13.7)

if st.button('Predict Rainfall'):
    input_data = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]], columns=feature_names)
    prediction = model.predict(input_data)[0]

    st.subheader('Prediction Result')
    if prediction == 1:
        st.success('🌧️ Rainfall Expected')
    else:
        st.info('☀️ No Rainfall Expected')

st.markdown('---')
st.caption('Made with Streamlit + Machine Learning')
