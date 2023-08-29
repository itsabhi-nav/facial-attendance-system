import streamlit as st
from home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title='Registration Form')
st.subheader('Registration Form')

# Initialise Registration form

registration_form =face_rec.RegistrationForm()

#STEP-1: Collect person name and role 
#form

person_name= st.text_input(label='Name', placeholder='First & Last Name')
role = st.selectbox(label='Select Your Role', options=('Student', 'Teacher'))

#Step-2: Collect facial embedding of that person

def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24')   #3D array bgr
    reg_img, embedding = registration_form.get_embedding(img)

# Two -Step process
# 1st step save data into local computer txt

    if embedding is not None:
        with open('face_embedding.txt', mode ='ab') as f:
            np.savetxt(f,embedding)

    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')

webrtc_streamer(key='registration',video_frame_callback=video_callback_func)

# Step-3: Save the data in redis database


if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(person_name,role)
    if return_val == True:
        st.success(f"{person_name} registered successfully")
    elif return_val == 'name_false':
        st.error('Please enter the name: Name cannot be empty or spaces')

    elif return_val == 'file_false':
        st.error('face_embedding.txt is not found. Please refresh the page and execute again.')


