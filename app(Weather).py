import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta

st.set_page_config(page_title='Weather Forecast ML', page_icon='☁️', layout='centered')

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    with open('rainfall_prediction_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['feature_names']

try:
    model, feature_names = load_model()
except Exception as e:
    st.error(f'Model file not found or invalid: {e}')
    st.stop()

# ---------- STYLE ----------
st.markdown('''
<style>
.stApp {
    background: linear-gradient(180deg,#5d6770,#1f2b34,#10151b);
    background-attachment: fixed;
}

.metric-pill {
    background: rgba(7,30,40,.78);
    padding: 12px 16px;
    border-radius: 999px;
    color: white;
    text-align:center;
}
.small-card {
    background: rgba(255,255,255,0.12);
    padding: 10px;
    border-radius: 18px;
    text-align:center;
}
.big {font-size:34px;font-weight:700;color:white;}
.mid {font-size:20px;color:white;}
.dim {color:#dfe7ef;}
label,p,h1,h2,h3,h4,h5,h6,div {color:white !important;}
</style>
''', unsafe_allow_html=True)

# ---------- SIDEBAR INPUTS ----------
st.sidebar.header('Weather Inputs')
pressure = st.sidebar.slider('Pressure', 980.0, 1050.0, 1015.9)
dewpoint = st.sidebar.slider('Dew Point', 0.0, 35.0, 19.9)
humidity = st.sidebar.slider('Humidity', 0, 100, 73)
cloud = st.sidebar.slider('Cloud Cover', 0, 100, 65)
sunshine = st.sidebar.slider('Sunshine Hours', 0.0, 12.0, 4.0)
winddirection = st.sidebar.slider('Wind Direction', 0.0, 360.0, 40.0)
windspeed = st.sidebar.slider('Wind Speed', 0.0, 60.0, 10.0)
city = st.sidebar.text_input('City', 'Kyiv, Ukraine')

input_df = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]], columns=feature_names)
pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1] if hasattr(model,'predict_proba') else 0.5

condition = 'Rainy' if pred == 1 else ('Cloudy' if cloud > 55 else 'Sunny')
emoji = '🌧️' if pred == 1 else ('☁️' if cloud > 55 else '☀️')
temp = round((dewpoint + (100-humidity)/8),1)
now = datetime.now()
sunrise = '4:53 am'
sunset = '8:13 pm'

# ---------- UI ----------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown(f"<div class='dim'>{now.strftime('%A, %H:%M')}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='big'>{emoji} {condition} {temp}°C</div>", unsafe_allow_html=True)
st.markdown(f"<div class='mid'>{city}</div>", unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='metric-pill'>🌅 {sunrise}</div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-pill'>☀️ 15 h 32 m</div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-pill'>🌇 {sunset}</div>", unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)
st.markdown(f"<div class='metric-pill'>☔ Rain chance: {int(prob*100)}%</div>", unsafe_allow_html=True)

c4,c5 = st.columns(2)
with c4:
    st.write(f'**Humidity:** {humidity}%')
with c5:
    st.write(f'**Wind:** {windspeed} km/h')

st.write('### 7-Day Forecast')
cols = st.columns(7)
icons = ['☁️','⛅','🌧️','☀️','☁️','🌧️','⛅']
for i,col in enumerate(cols):
    day = (now + timedelta(days=i)).strftime('%a')
    with col:
        st.markdown(f"<div class='small-card'><b>{day}</b><br>{icons[i]}<br>{temp+i%3:.0f}°<br>{max(temp-4,1):.0f}°</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

