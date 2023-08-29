import streamlit as st


st.set_page_config(page_title='Attendance System', layout='wide')

st.header('Attendance System Using face Recognition')

with st.spinner("Loading Models and connecting to Redis Database....."):
    import face_rec

st.success('Model loaded successfully')
st.success('Redis Database Successfully Connected')