import streamlit as st
import face_rec  # Import your face_rec module

from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title='Real-Time Attendance System')
st.subheader('Real-Time Attendance System')

# Retrieve the data from Redis Database
with st.spinner('Retrieving Data from Redis Database...'):
    redis_face_db = face_rec.retrive_data(name='academy:register')
    st.dataframe(redis_face_db)
    
st.success("Data successfully retrieved from the Database")


#Time
waitTime =30 #time in second
setTime = time.time()
realtimepred = face_rec.RealTimePred()  # Corrected the assignment operator
 # real time prediction class


# Real Time Prediction
# Streamlit WebRTC

# callback function
def video_frame_callback(frame):
    global setTime

    img = frame.to_ndarray(format="bgr24")  #bgr 24 is 3 dimension numpy array
    # operation that you can perform on the array
    pred_img = realtimepred.face_prediction(img,redis_face_db,
                                        'facial_features',['Name','Role'],thresh=0.5)
    
    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time()   # reset time

        print('Save Data To Redis Database')


    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)



