#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import cv2
import mediapipe as mp
import time
import os
import textwrap

# =============================
# Global Variables
# =============================
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)


# =================
# WebRTC integration
#==================
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # Initialize your gesture recognition model here
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        # Add your gesture processing logic here
        return image

webrtc_streamer(key="gesture-recognition", video_processor_factory=VideoProcessor)


dot_positions = []
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
ANIMATION_PATH_ROOT = r'C:\Users\Administrator\Desktop\python\animations'

TARGET_FPS = 30
FRAME_DURATION = 1.0 / TARGET_FPS

gesture_timers = [None] * 6
sustain_duration = 1  # <-- 0.4 seconds hover required
animation_frames = []
frame_index = 0
animation_playing = False
last_triggered_dot = -1
last_animation_end_time = 0
ANIMATION_COOLDOWN = 1.0  # seconds

dot_opacity = 1.0
FADE_DURATION = 0.5  # seconds
fade_state = "idle"
fade_start_time = None

SCREEN_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
SCREEN_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Button state
show_tracking_markers = True
show_feedback = True
frame_height_global = 0
frame_width_global = 0

# Feedback
feedback_message = ""
feedback_timer = None
feedback_duration = 2.5

# =============================
# Helper Functions
# =============================
def get_animation_path(color_name, gesture_name):
    return os.path.join(ANIMATION_PATH_ROOT, f"{color_name}_{gesture_name}")

def load_animation_frames(color_name, gesture_name):
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
        h, w = frame.shape[:2]
        scale_ratio = min(SCREEN_WIDTH / w, SCREEN_HEIGHT / h)
        new_w = int(w * scale_ratio)
        new_h = int(h * scale_ratio)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        padded = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8)
        x_pad = (SCREEN_WIDTH - new_w) // 2
        y_pad = (SCREEN_HEIGHT - new_h) // 2
        padded[y_pad:y_pad+new_h, x_pad:x_pad+new_w] = resized
        frames.append(padded)
    if not frames:
        return None, f"Failed to load images from: {path}"
    return frames, None

def draw_circle_of_dots(image_size=(720, 640), center=(360, 480), radius=100, dot_radius=25):
    image = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)
    dot_positions.clear()
    for i, color in enumerate(dot_colors):
        angle = np.pi / 2 - (2 * np.pi * i / 6)
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] - radius * np.sin(angle))
        dot_positions.append((x, y - 30))
        cv2.circle(image, (x, y - 30), dot_radius, color + (255,), -1)
    return image

dot_image = draw_circle_of_dots()

def draw_wrapped_text(img, text, pos_y, font, font_scale, color, thickness, max_width):
    wrapper = textwrap.TextWrapper(width=max_width)
    lines = wrapper.wrap(text)
    y = pos_y
    line_height = int(font_scale * 40)
    for line in lines:
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = int((img.shape[1] - text_size[0]) / 2)
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height

# =============================
# Mouse Callback for Buttons
# =============================
def mouse_callback(event, x, y, flags, param):
    global show_tracking_markers, show_feedback, feedback_message, feedback_timer
    button_height = 40
    button_width = 150
    button_y = frame_height_global - 60

    # Tracking markers button (bottom left)
    tracking_button_x = 20
    if (tracking_button_x < x < tracking_button_x + button_width and 
        button_y < y < button_y + button_height and 
        event == cv2.EVENT_LBUTTONDOWN):
        show_tracking_markers = not show_tracking_markers

    # Feedback button (bottom right)
    feedback_button_x = frame_width_global - 20 - button_width
    if (feedback_button_x < x < feedback_button_x + button_width and 
        button_y < y < button_y + button_height and 
        event == cv2.EVENT_LBUTTONDOWN):
        show_feedback = not show_feedback
        if not show_feedback:
            feedback_message = ""
            feedback_timer = None

cv2.namedWindow('Hand Tracking')
cv2.setMouseCallback('Hand Tracking', mouse_callback)

# =============================
# Gesture Detection Functions
# =============================
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
# Main Loop
# =============================
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_height_global = frame.shape[0]
        frame_width_global = frame.shape[1]

        # Flip and convert frame
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        webcam_feed = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_time = time.time()

        # --- Dot fade logic ---
        if fade_state == "idle":
            dot_opacity = 0.6 
        if animation_playing and fade_state == "idle":
            fade_state = "fading_out"
            fade_start_time = current_time
        if fade_state == "fading_out":
            elapsed = current_time - fade_start_time
            dot_opacity = max(0.0, 0.6 - (elapsed / FADE_DURATION))
            if dot_opacity <= 0.0:
                dot_opacity = 0.0
                fade_state = "faded"
        if not animation_playing and fade_state == "faded":
            fade_state = "fading_in"
            fade_start_time = current_time
        if fade_state == "fading_in":
            elapsed = current_time - fade_start_time
            dot_opacity = min(0.6 , elapsed / FADE_DURATION)
            if dot_opacity >= 0.6 :
                dot_opacity = 0.6 
                fade_state = "idle"

        # Draw dots with current opacity
        if dot_opacity > 0.0:
            scale_x = frame_width_global / dot_image.shape[1]
            scale_y = frame_height_global / dot_image.shape[0]
            scale = min(scale_x, scale_y)
            new_width = int(dot_image.shape[1] * scale)
            new_height = int(dot_image.shape[0] * scale)
            dot_overlay_resized = cv2.resize(dot_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            dot_overlay_resized = dot_overlay_resized.copy()
            dot_overlay_resized[:, :, 3] = (dot_overlay_resized[:, :, 3].astype(np.float32) * dot_opacity).astype(np.uint8)
            x_offset = (frame_width_global - new_width) // 2
            y_offset = (frame_height_global - new_height) // 2
            overlay = np.zeros((frame_height_global, frame_width_global, 4), dtype=np.uint8)
            overlay[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = dot_overlay_resized
            webcam_feed_rgba = cv2.cvtColor(webcam_feed, cv2.COLOR_BGR2BGRA)
            alpha_overlay = overlay[:, :, 3:4] / 255.0
            webcam_feed_rgba = (webcam_feed_rgba * (1 - alpha_overlay) + overlay * alpha_overlay).astype(np.uint8)
            webcam_feed = cv2.cvtColor(webcam_feed_rgba, cv2.COLOR_BGRA2BGR)

        # Detect gestures and provide contextual feedback
        detected_gestures = [False] * 6
        gesture_context = None  # (dot_index, gesture_name)
        if results.multi_hand_landmarks and dot_opacity > 0.5:
            # Get hand position
            if len(results.multi_hand_landmarks) == 2:
                hand1, hand2 = results.multi_hand_landmarks
                index_tip_1 = hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_tip_2 = hand2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int((index_tip_1.x + index_tip_2.x) / 2 * frame_width_global)
                y = int((index_tip_1.y + index_tip_2.y) / 2 * frame_height_global)
            else:
                index_tip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_tip.x * frame_width_global)
                y = int(index_tip.y * frame_height_global)

            scale = min(frame_width_global / dot_image.shape[1], frame_height_global / dot_image.shape[0])
            x_offset = (frame_width_global - int(dot_image.shape[1] * scale)) // 2
            y_offset = (frame_height_global - int(dot_image.shape[0] * scale)) // 2

            for i, (dot_x, dot_y) in enumerate(dot_positions):
                scaled_dot_x = int(dot_x * scale) + x_offset
                scaled_dot_y = int(dot_y * scale) + y_offset
                distance = np.hypot(x - scaled_dot_x, y - scaled_dot_y)
                if distance < 30:
                    if gesture_timers[i] is None:
                        gesture_timers[i] = time.time()
                    elif time.time() - gesture_timers[i] >= sustain_duration:
                        # Determine which gesture is being performed
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
                            gesture_timers[i] = None
                    # Draw a ring in the color of the dot, with full opacity
                    cv2.circle(webcam_feed, (scaled_dot_x, scaled_dot_y), 30, dot_colors[i], 4)
                else:
                    gesture_timers[i] = None

            # Draw tracking markers
            if show_tracking_markers:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        webcam_feed, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 200, 200), circle_radius=5),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

        # Draw control buttons with dark gray background
        button_height = 40
        button_width = 150
        button_y = frame_height_global - 60
        button_color = (50, 50, 50)  # Dark gray

        # Tracking markers button (bottom left)
        tracking_button_x = 20
        cv2.rectangle(webcam_feed,
                     (tracking_button_x, button_y),
                     (tracking_button_x + button_width, button_y + button_height),
                     button_color, -1)
        tracking_text = "Tracking: ON" if show_tracking_markers else "Tracking: OFF"
        cv2.putText(webcam_feed, tracking_text,
                   (tracking_button_x + 10, button_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Feedback button (bottom right)
        feedback_button_x = frame_width_global - 20 - button_width
        cv2.rectangle(webcam_feed,
                     (feedback_button_x, button_y),
                     (feedback_button_x + button_width, button_y + button_height),
                     button_color, -1)
        feedback_text = "Feedback: ON" if show_feedback else "Feedback: OFF"
        cv2.putText(webcam_feed, feedback_text,
                   (feedback_button_x + 10, button_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Provide contextual feedback if gesture detected
        if gesture_context is not None:
            dot_idx, gesture_name = gesture_context
            color = color_names[dot_idx]
            gesture_friendly = gesture_friendly_names.get(gesture_name, gesture_name)
            # Contextual feedback
            if gesture_name == "Hand_Heart":
                msg = f"Beautiful! You made a {color} Hand Heart gesture!"
            elif gesture_name == "Finger_Heart":
                msg = f"Nice! {color} Finger Heart gesture detected!"
            elif gesture_name == "Middle_Finger":
                msg = f"Oops! That's a {color} Middle Finger gesture!"
            elif gesture_name == "Thumbs_Up":
                msg = f"Thumbs Up! {color} dot recognized your gesture!"
            elif gesture_name == "Thumbs_Down":
                msg = f"Thumbs Down on {color}! Gesture detected."
            else:
                msg = f"{color} dot: {gesture_friendly} gesture detected!"
            feedback_message = msg
            feedback_timer = current_time

        # Draw feedback message if toggled on
        if show_feedback and feedback_message and feedback_timer and (current_time - feedback_timer < feedback_duration):
            draw_wrapped_text(
                webcam_feed, feedback_message, 50,
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, max_width=32
            )

        # Play animation if active
        if animation_playing and animation_frames:
            if frame_index < len(animation_frames):
                frame_rgba = animation_frames[frame_index]
                frame_index += 1
                anim_alpha = frame_rgba[:, :, 3:4] / 255.0
                webcam_feed_rgba = cv2.cvtColor(webcam_feed, cv2.COLOR_BGR2BGRA)
                webcam_feed_rgba = (webcam_feed_rgba * (1 - anim_alpha) + frame_rgba * anim_alpha).astype(np.uint8)
                webcam_feed = cv2.cvtColor(webcam_feed_rgba, cv2.COLOR_BGRA2BGR)
            else:
                animation_playing = False
                animation_frames = []
                frame_index = 0
                last_animation_end_time = time.time()

        # Trigger animation
        for i, detected in enumerate(detected_gestures):
            if detected and not animation_playing:
                if time.time() - last_animation_end_time < ANIMATION_COOLDOWN:
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
                    animation_frames, error = load_animation_frames(color_name, gesture_name)
                    if animation_frames:
                        animation_playing = True
                        frame_index = 0
                        last_triggered_dot = i
                    else:
                        if show_feedback:
                            feedback_message = error
                            feedback_timer = time.time()

        cv2.imshow('Hand Tracking', webcam_feed)

        elapsed = time.time() - loop_start
        if elapsed < FRAME_DURATION:
            time.sleep(FRAME_DURATION - elapsed)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
get_ipython().run_line_magic('%writefile', 'myscript.py')
print("Hello from my script!")


# In[ ]:





# In[ ]:




