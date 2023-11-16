import streamlit as st
import cv2
from ultralytics import YOLO

# Load your YOLO model
model_path = '/Users/shrinathkhadake/Downloads/yolo-nas.pt'  # Update this path
model = YOLO(model_path)

# Class name dictionary
class_name_dict = {0: 'A1', 1: 'SC', 2: 'PCS', 3: 'SP', 4: 'CKS', 5: 'P', 6: 'FG/B', 7: 'FG/W', 8: 'CH', 9: 'WR', 10: 'S6/MM', 11: '7B/BS', 12: 'RSNL', 13: 'CSNL', 14: 'LSC', 15: 'FMP', 16: 'ID'}

# Set the app layout
st.set_page_config(layout="wide")

# Initialize session state
if 'is_running' not in st.session_state:
    st.session_state.is_running = True

# Divide the screen into two parts (50% each)
col1, col2 = st.columns(2)

# Upper part (Camera feed and Object Detection)
with col1:
    st.header("Camera Feed and Object Detection")
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

            # Crop the frame to 640x640
            height, width, _ = frame.shape
            start_x = max(0, (width - 640) // 2)
            start_y = max(0, (height - 640) // 2)
            cropped_frame = frame[start_y:start_y + 640, start_x:start_x + 640]

            # Perform object detection using the YOLO model
            results = model(cropped_frame)[0]
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > 0.2:  # Threshold
                    cv2.rectangle(cropped_frame, (int(x1), int(y1)), (int(x2), int(y2)),(0, 255, 0), thickness=1)
                    cv2.putText(cropped_frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB color space
            video_placeholder.image(cropped_frame, width=640)  # Display cropped and processed frame

# Add a button to stop the video stream
with col2:
    stop_button = st.button("Stop Video Stream")

    if stop_button:
        st.session_state.is_running = False
        cap.release()

