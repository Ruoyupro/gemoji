import streamlit as st
from main import gesture_recognition_logic  # Import your core logic

def main():
    st.title("Interactive Gesture Control")
    st.markdown("### Webcam Gesture Recognition")
    
    # Add WebRTC component
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
    
    # Add UI elements from your original code
    st.sidebar.checkbox("Show Tracking Markers", value=True)
    st.sidebar.checkbox("Enable Feedback", value=True)

if __name__ == "__main__":
    main()
