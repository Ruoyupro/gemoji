#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import numpy as np
import cv2
import mediapipe as mp
import os

# =============================
# Global Variables & Constants
# =============================
dot_colors = [
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (0, 255, 0),    # Green
    (255, 255, 255),# White
    (255, 0, 0),    # Blue
    (255, 0, 255)   # Purple
]
color_names = ['Red', 'Yellow', 'Green', 'White', 'Blue', 'Purple']
gesture_names = ['Hand_Heart', 'Finger_Heart', 'Middle_Finger', 'Thumbs_Up', 'Thumbs_Down']
gesture_friendly_names = {
    'Hand_Heart': 'Hand Heart',
    'Finger_Heart': 'Finger Heart',
    'Middle_Finger': 'Middle Finger',
    'Thumbs_Up': 'Thumbs Up',
    'Thumbs_Down': 'Thumbs Down'
}

ANIMATION_PATH_ROOT = os.path.join(os.getcwd(), "animations")
FRAME_SIZE = (640, 480)

# =============================
# Helper Functions
# =============================
def get_animation_path(color_name, gesture_name):
    return os.path.join(ANIMATION_PATH_ROOT, f"{color_name}_{gesture_name}")

def load_animation_frames(color_name, gesture_name, frame_size):
    path = get_animation_path(color_name, gesture_name)
    if not os.path.exists(path):
        return None
    files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()
    frames = []
    for filename in files:
        frame_path = os.path.join(path, filename)
        frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            continue
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
    return frames if frames else None

# =============================
# Gesture Detection Functions
# =============================
mp_hands = mp.solutions.hands

def is_finger_closed(hand_landmarks, finger_tip, finger_pip):
    tip = hand_landmarks.landmark[finger_tip]
    pip = hand_landmarks.landmark[finger_pip]
    return tip.y > pip.y

def is_hand_upright(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    return wrist.y > mcp.y

def is_heart_gesture(hands_list):
    if len(hands_list) != 2:
        return False
    hand1, hand2 = hands_list
    if not (is_hand_upright(hand1) and is_hand_upright(hand2)):
        return False
    for hand in hands_list:
        if not (
            is_finger_closed(hand, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP) and
            is_finger_closed(hand, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP) and
            is_finger_closed(hand, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
        ):
            return False
    thumb1 = hand1.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index1 = hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb2 = hand2.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index2 = hand2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_dist = np.hypot(thumb1.x - thumb2.x, thumb1.y - thumb2.y)
    index_dist = np.hypot(index1.x - index2.x, index1.y - index2.y)
    return thumb_dist < 0.15 and index_dist < 0.15

def is_finger_heart(hand_landmarks):
    if not is_hand_upright(hand_landmarks):
        return False
    if not all(
        is_finger_closed(hand_landmarks, tip, pip) for tip, pip in [
            (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
            (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
        ]
    ):
        return False
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.hypot(thumb.x - index.x, thumb.y - index.y)
    return distance < 0.05

def is_middle_finger(hand_landmarks):
    if not (
        is_finger_closed(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP) and
        is_finger_closed(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP) and
        is_finger_closed(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ):
        return False
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    return middle_tip.y < middle_pip.y

def is_thumbs_up(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_up = (thumb_tip.y < wrist.y - 0.02 and
                thumb_tip.y < thumb_ip.y - 0.01 and
                thumb_ip.y < thumb_mcp.y + 0.03)
    open_fingers = 0
    for tip, pip in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]:
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[pip].y
        if tip_y < pip_y - 0.05:
            open_fingers += 1
    return thumb_up and open_fingers <= 1

def is_thumbs_down(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_down = (thumb_tip.y > wrist.y + 0.02 and
                  thumb_tip.y > thumb_ip.y + 0.01 and
                  thumb_ip.y > thumb_mcp.y - 0.03)
    open_fingers = 0
    for tip, pip in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]:
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[pip].y
        if tip_y > pip_y + 0.05:
            open_fingers += 1
    return thumb_down and open_fingers <= 1

# =============================
# Streamlit WebRTC Video Processor
# =============================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_size = FRAME_SIZE
        self.color_name = "Red"
        self.gesture_name = "Hand_Heart"
        self.animations = {}
        for gesture in gesture_names:
            self.animations[gesture] = load_animation_frames(self.color_name, gesture, self.frame_size)
        self.anim_index = 0
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        detected_gesture = None
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 2 and is_heart_gesture(results.multi_hand_landmarks):
                detected_gesture = "Hand_Heart"
            elif len(results.multi_hand_landmarks) == 1:
                hand = results.multi_hand_landmarks[0]
                if is_finger_heart(hand):
                    detected_gesture = "Finger_Heart"
                elif is_middle_finger(hand):
                    detected_gesture = "Middle_Finger"
                elif is_thumbs_up(hand):
                    detected_gesture = "Thumbs_Up"
                elif is_thumbs_down(hand):
                    detected_gesture = "Thumbs_Down"

            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Overlay animation for detected gesture
        if detected_gesture and self.animations[detected_gesture]:
            frames = self.animations[detected_gesture]
            anim = frames[self.anim_index % len(frames)]
            alpha_s = anim[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                img[:, :, c] = (alpha_s * anim[:, :, c] +
                                alpha_l * img[:, :, c])
            self.anim_index += 1
        elif self.animations[self.gesture_name]:
            # Default animation overlay (for demo)
            anim = self.animations[self.gesture_name][self.anim_index % len(self.animations[self.gesture_name])]
            alpha_s = anim[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                img[:, :, c] = (alpha_s * anim[:, :, c] +
                                alpha_l * img[:, :, c])
            self.anim_index += 1

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =============================
# Streamlit UI
# =============================
st.title("Gemoji Gesture Animation Demo")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="gemoji-demo",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

st.info("Show your hand gesture to see the animation overlay! (Gesture detection is implemented for demo gestures.)")


# In[ ]:




