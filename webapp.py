import streamlit as st
import cv2
import pandas as pd
from PIL import Image
import numpy as np

# Create a sample DataFrame for the table
data = {'Column 1': ['Data 1', 'Data 2'],
        'Column 2': ['Data A', 'Data B']}
df = pd.DataFrame(data)

# Set the app layout
st.set_page_config(layout="wide")

# Initialize session state
if 'is_running' not in st.session_state:
    st.session_state.is_running = True

# Divide the screen into two parts (50% each)
col1, col2 = st.columns(2)

# Lower part (Table)
with col2:
    st.header("Table")
    st.table(df)

# Upper part (Camera feed)
with col1:
    st.header("Camera Feed")
    # Use OpenCV to capture the camera feed (replace 0 with the camera index if necessary)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.warning("Unable to access the camera. Please check your camera connection.")
    else:
        # Create a placeholder for the video stream
        video_placeholder = st.empty()

        # Continuously display the camera stream
        while st.session_state.is_running:
            ret, frame = cap.read()

            if not ret:
                st.warning("Failed to capture camera feed. Please check your camera connection.")
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to JPEG
            frame_jpeg = Image.fromarray(rgb_frame).convert("RGB")
            frame_jpeg = np.array(frame_jpeg)

            # Display live video stream
            video_placeholder.image([frame_jpeg], channels="RGB", use_column_width=True)
            
           # st.table(df)

# Lower part (Table)
with col2:
    st.header("Table")
    st.table(df)

# Add a button to stop the video stream
stop_button = st.button("Stop Video Stream")

if stop_button:
    st.session_state.is_running = False
    cap.release()

