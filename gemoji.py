import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import numpy as np
import mediapipe as mp
import os
import time
import textwrap
from typing import List, Dict, Optional, Tuple

# =============================
# Handle OpenCV Import Safely
# =============================
try:
    import cv2
except ImportError as e:
    st.error("""
        Failed to import OpenCV (cv2). This usually happens because:
        - Missing system libraries (like libgl1-mesa-dri)
        - Incorrect OpenCV package installed
        - Conflicting installations
        
        Please ensure:
        - 'opencv-contrib-python' is used instead of 'opencv-python'
        - 'packages.txt' contains: libgl1-mesa-dri, libglib2.0-0, and libglx0
    """)
    st.code(str(e))
    st.stop()

# =============================
# Prompt User Before Downloading Animations
# =============================
ANIMATION_REPO_PATH = "gemoji"
ANIMATION_GITHUB_URL = "https://github.com/Ruoyupro/gemoji.git"

if not os.path.exists(ANIMATION_REPO_PATH):
    st.warning("This app needs to download animations (~50MB). Do you want to proceed?")
    proceed = st.checkbox("Yes, download animations")
    if not proceed:
        st.stop()
    else:
        status_box = st.info("Cloning animation repository from GitHub... Please wait.")
        import subprocess
        result = subprocess.run([
            "git", "clone", ANIMATION_GITHUB_URL, ANIMATION_REPO_PATH
        ], capture_output=True, text=True)
        if result.returncode != 0:
            status_box.error("Failed to download animations.")
            st.code(result.stderr)
            st.stop()
        else:
            status_box.success("Animations downloaded successfully.")

# Set animation path root
ANIMATION_PATH_ROOT = os.path.join(ANIMATION_REPO_PATH, "animations")

# =============================
# Global Variables & Constants
# =============================
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

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
FRAME_SIZE = (320, 240)
TARGET_FPS = 30
FRAME_DURATION = 1.0 / TARGET_FPS
sustain_duration = 1  # seconds hover required
ANIMATION_COOLDOWN = 1.0  # seconds
FADE_DURATION = 0.5  # seconds

# =============================
# Helper Functions
# =============================
def get_animation_path(color_name, gesture_name):
    return os.path.join(ANIMATION_PATH_ROOT, f"{color_name}_{gesture_name}")

def load_animation_frames(color_name, gesture_name, frame_size):
    path = get_animation_path(color_name, gesture_name)
    if not os.path.exists(path):
        return None, f"Animation folder not found: {path}"
    files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        return None, f"No image files found in: {path}"
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
    if not frames:
        return None, f"Failed to load images from: {path}"
    return frames, None

def draw_circle_of_dots(image_size=(320, 240), center=(160, 120), radius=50, dot_radius=15):
    image = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)
    dot_positions = []
    for i, color in enumerate(dot_colors):
        angle = np.pi / 2 - (2 * np.pi * i / 6)
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] - radius * np.sin(angle))
        dot_positions.append((x, y - 15))
        cv2.circle(image, (x, y - 15), dot_radius, color + (255,), -1)
    return image, dot_positions

def draw_wrapped_text(img, text, pos_y, font, font_scale, color, thickness, max_width):
    wrapper = textwrap.TextWrapper(width=max_width)
    lines = wrapper.wrap(text)
    y = pos_y
    line_height = int(font_scale * 20)
    for line in lines:
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = int((img.shape[1] - text_size[0]) / 2)
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height

# =============================
# Gesture Detection Logic
# =============================
def is_heart_gesture(hands_landmarks):
    if len(hands_landmarks) != 2:
        return False
    hand1 = hands_landmarks[0].landmark
    hand2 = hands_landmarks[1].landmark
    thumb_tip1 = hand1[mp_hands.HandLandmark.THUMB_TIP]
    index_tip1 = hand1[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip2 = hand2[mp_hands.HandLandmark.THUMB_TIP]
    index_tip2 = hand2[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    dist_thumb_tips = ((thumb_tip1.x - thumb_tip2.x)**2 + (thumb_tip1.y - thumb_tip2.y)**2)**0.5
    dist_index_tips = ((index_tip1.x - index_tip2.x)**2 + (index_tip1.y - index_tip2.y)**2)**0.5
    return dist_thumb_tips < 0.05 and dist_index_tips < 0.05

def is_finger_heart(hand_landmarks):
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    return (
        thumb_tip.x > index_tip.x and
        abs(pinky_tip.y - middle_tip.y) < 0.05 and
        abs(ring_tip.y - middle_tip.y) < 0.05
    )

def is_middle_finger(hand_landmarks):
    landmarks = hand_landmarks.landmark
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    return middle_tip.y < wrist.y

def is_thumbs_up(hand_landmarks):
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    return thumb_tip.y < wrist.y

def is_thumbs_down(hand_landmarks):
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    return thumb_tip.y > wrist.y

# =============================
# Streamlit Video Processor
# =============================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_size = FRAME_SIZE
        self.dot_image, self.dot_positions = draw_circle_of_dots(self.frame_size)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.gesture_timers = [None] * 6
        self.animation_frames = []
        self.frame_index = 0
        self.animation_playing = False
        self.last_animation_end_time = 0
        self.last_triggered_dot = -1
        self.dot_opacity = 0.6
        self.fade_state = "idle"
        self.fade_start_time = None
        self.feedback_message = ""
        self.feedback_timer = None
        self.feedback_duration = 2.5
        self.show_tracking_markers = True
        self.show_feedback = True

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
        except Exception as e:
            return av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), dtype=np.uint8), format="bgr24")

        current_time = time.time()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        # --- Dot fade logic ---
        if self.fade_state == "idle":
            self.dot_opacity = 0.6
        if self.animation_playing and self.fade_state == "idle":
            self.fade_state = "fading_out"
            self.fade_start_time = current_time
        if self.fade_state == "fading_out":
            elapsed = current_time - self.fade_start_time
            self.dot_opacity = max(0.0, 0.6 - (elapsed / FADE_DURATION))
            if self.dot_opacity <= 0.0:
                self.dot_opacity = 0.0
                self.fade_state = "faded"
        if not self.animation_playing and self.fade_state == "faded":
            self.fade_state = "fading_in"
            self.fade_start_time = current_time
        if self.fade_state == "fading_in":
            elapsed = current_time - self.fade_start_time
            self.dot_opacity = min(0.6, elapsed / FADE_DURATION)
            if self.dot_opacity >= 0.6:
                self.dot_opacity = 0.6
                self.fade_state = "idle"

        # Draw dots with current opacity
        if self.dot_opacity > 0.0:
            dot_overlay = self.dot_image.copy()
            dot_overlay[:, :, 3] = (dot_overlay[:, :, 3].astype(np.float32) * self.dot_opacity).astype(np.uint8)
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            alpha_overlay = dot_overlay[:, :, 3:4] / 255.0
            img_rgba = (img_rgba * (1 - alpha_overlay) + dot_overlay * alpha_overlay).astype(np.uint8)
            img = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)

        detected_gestures = [False] * 6
        gesture_context = None

        if results.multi_hand_landmarks and self.dot_opacity > 0.5:
            # Get index finger tip position
            if len(results.multi_hand_landmarks) == 2:
                hand1, hand2 = results.multi_hand_landmarks
                index_tip_1 = hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_tip_2 = hand2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int((index_tip_1.x + index_tip_2.x) / 2 * self.frame_size[0])
                y = int((index_tip_1.y + index_tip_2.y) / 2 * self.frame_size[1])
            else:
                index_tip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_tip.x * self.frame_size[0])
                y = int(index_tip.y * self.frame_size[1])

            for i, (dot_x, dot_y) in enumerate(self.dot_positions):
                distance = np.hypot(x - dot_x, y - dot_y)
                if distance < 30:
                    if self.gesture_timers[i] is None:
                        self.gesture_timers[i] = current_time
                    elif current_time - self.gesture_timers[i] >= sustain_duration:
                        gesture_name = ""
                        if len(results.multi_hand_landmarks) == 2:
                            if is_heart_gesture(results.multi_hand_landmarks):
                                gesture_name = "Hand_Heart"
                        else:
                            for hand in results.multi_hand_landmarks:
                                if is_finger_heart(hand):
                                    gesture_name = "Finger_Heart"
                                elif is_middle_finger(hand):
                                    gesture_name = "Middle_Finger"
                                elif is_thumbs_up(hand):
                                    gesture_name = "Thumbs_Up"
                                elif is_thumbs_down(hand):
                                    gesture_name = "Thumbs_Down"
                        if gesture_name:
                            detected_gestures[i] = True
                            gesture_context = (i, gesture_name)
                            self.gesture_timers[i] = None
                    # Draw ring around hovered dot
                    cv2.circle(img, (dot_x, dot_y), 30, dot_colors[i], 4)
                else:
                    self.gesture_timers[i] = None

            # Draw tracking markers
            if self.show_tracking_markers:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 200, 200), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

        # Provide contextual feedback if gesture detected
        if gesture_context is not None:
            dot_idx, gesture_name = gesture_context
            color = color_names[dot_idx]
            gesture_friendly = gesture_friendly_names.get(gesture_name, gesture_name)
            msg = f"{color} dot: {gesture_friendly} gesture detected!"
            self.feedback_message = msg
            self.feedback_timer = current_time

        # Draw feedback message if toggled on
        if self.show_feedback and self.feedback_message and (current_time - self.feedback_timer < self.feedback_duration):
            draw_wrapped_text(
                img, self.feedback_message, 50,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, max_width=32
            )

        # Play animation if active
        if self.animation_playing and self.animation_frames:
            if self.frame_index < len(self.animation_frames):
                frame_rgba = self.animation_frames[self.frame_index]
                self.frame_index += 1
                anim_alpha = frame_rgba[:, :, 3:4] / 255.0
                img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                img_rgba = (img_rgba * (1 - anim_alpha) + frame_rgba * anim_alpha).astype(np.uint8)
                img = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
            else:
                self.animation_playing = False
                self.animation_frames = []
                self.frame_index = 0
                self.last_animation_end_time = current_time

        # Trigger animation
        for i, detected in enumerate(detected_gestures):
            if detected and not self.animation_playing:
                if current_time - self.last_animation_end_time < ANIMATION_COOLDOWN:
                    continue
                gesture_name = ""
                if results.multi_hand_landmarks:
                    if len(results.multi_hand_landmarks) == 2:
                        if is_heart_gesture(results.multi_hand_landmarks):
                            gesture_name = "Hand_Heart"
                    else:
                        for hand in results.multi_hand_landmarks:
                            if is_finger_heart(hand):
                                gesture_name = "Finger_Heart"
                            elif is_middle_finger(hand):
                                gesture_name = "Middle_Finger"
                            elif is_thumbs_up(hand):
                                gesture_name = "Thumbs_Up"
                            elif is_thumbs_down(hand):
                                gesture_name = "Thumbs_Down"
                if gesture_name:
                    color_name = color_names[i]
                    animation_frames, error = load_animation_frames(color_name, gesture_name, self.frame_size)
                    if animation_frames:
                        self.animation_playing = True
                        self.animation_frames = animation_frames
                        self.frame_index = 0
                        self.last_triggered_dot = i
                    else:
                        if self.show_feedback:
                            self.feedback_message = error
                            self.feedback_timer = current_time

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =============================
# Streamlit UI
# =============================
st.title("Gemoji Gesture Control")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    show_tracking = st.checkbox("Show Hand Tracking", value=True)
    show_feedback = st.checkbox("Show Feedback Messages", value=True)
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Make gestures with your hands")
    st.markdown("2. Hover over a colored dot for 1 second")
    st.markdown("3. Perform one of the supported gestures")
    st.markdown("4. See the animation play!")
    st.markdown("---")
    st.markdown("**Supported Gestures:**")
    st.markdown("- Hand Heart (two hands)")
    st.markdown("- Finger Heart")
    st.markdown("- Thumbs Up")
    st.markdown("- Thumbs Down")
    st.markdown("- Middle Finger")

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Create the video streamer
ctx = webrtc_streamer(
    key="gemoji",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# Update settings
if ctx.video_processor:
    ctx.video_processor.show_tracking_markers = show_tracking
    ctx.video_processor.show_feedback = show_feedback