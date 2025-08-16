# import av
# import cv2
# import mediapipe as mp
# import numpy as np
# import tflite_runtime.interpreter as tflite
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, AudioProcessorBase, RTCConfiguration
# import logging
# import asyncio

# # ---------------- CONFIG ----------------
# IMG_WIDTH, IMG_HEIGHT = 224, 224
# MODEL_PATH = "model/drowsiness_model.tflite"
# MOUTH_ROI_SCALE = 1.5
# PADDING = 10
# VIDEO_FPS = 20
# EYE_AR_CONSEC_FRAMES = 3
# FATIGUE_COOLDOWN = 5.0

# # Setup logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # ---------------- Load TFLite ----------------
# try:
#     interpreter = tflite.Interpreter(model_path=MODEL_PATH)
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     logger.info("TFLite model loaded successfully")
# except Exception as e:
#     st.error(f"Failed to load TFLite model: {e}")
#     logger.error(f"Model loading error: {e}")
#     st.stop()

# # ---------------- Helper Functions ----------------
# def preprocess_image(img):
#     img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT))
#     img = img.astype(np.float32) / 255.0
#     return np.expand_dims(img, axis=0)

# def extract_roi(image, landmarks, points, is_mouth=False, img_width=None, img_height=None):
#     x_coords = [landmarks[i].x * img_width for i in points]
#     y_coords = [landmarks[i].y * img_height for i in points]
#     x_min, x_max = int(min(x_coords)), int(max(x_coords))
#     y_min, y_max = int(min(y_coords)), int(max(y_coords))

#     if is_mouth:
#         width = x_max - x_min
#         height = y_max - y_min
#         x_min = max(0, x_min - int(width * (MOUTH_ROI_SCALE - 1) / 2))
#         x_max = min(image.shape[1], x_max + int(width * (MOUTH_ROI_SCALE - 1) / 2))
#         y_min = max(0, y_min - int(height * (MOUTH_ROI_SCALE - 1) / 2))
#         y_max = min(image.shape[0], y_max + int(height * (MOUTH_ROI_SCALE - 1) / 2))
#     else:
#         x_min = max(0, x_min - PADDING)
#         x_max = min(image.shape[1], x_max + PADDING)
#         y_min = max(0, y_min - PADDING)
#         y_max = min(image.shape[0], y_max + PADDING)

#     roi = image[y_min:y_max, x_min:x_max]
#     if roi.size == 0:
#         logger.warning("Empty ROI detected")
#         return None, None
#     return roi, (x_min, y_min, x_max, y_max)

# def predict_drowsiness(roi, is_eye=True):
#     if roi is None:
#         return None
#     try:
#         input_data = preprocess_image(roi)
#         interpreter.set_tensor(input_details[0]['index'], input_data)
#         interpreter.invoke()
#         prediction = interpreter.get_tensor(output_details[0]['index'])[0]
#         if is_eye:
#             return np.argmax(prediction[:2]), prediction[:2]  # 0=Closed, 1=Open
#         else:
#             return np.argmax(prediction[2:]) + 2, prediction[2:]  # 2=NoYawn, 3=Yawn
#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#         return None

# # ---------------- Audio Alarm ----------------
# class AlarmAudioProcessor(AudioProcessorBase):
#     def __init__(self):
#         self.fatigued = False
#         self.last_alert_time = 0

#     async def recv_queued(self, frames: list[av.AudioFrame]) -> list[av.AudioFrame]:
#         if not frames:
#             return frames
        
#         current_time = asyncio.get_event_loop().time()
#         if self.fatigued and (current_time - self.last_alert_time) > 1.0:
#             sr = 48000
#             samples = np.arange(sr)  # 1-second tone
#             tone = (np.sin(2 * np.pi * 440 * samples / sr) * 0.7).astype(np.float32)  # Increased volume to 0.7
#             aframe = av.AudioFrame.from_ndarray(tone, layout="mono")
#             aframe.sample_rate = sr
#             logger.debug("Playing fatigue alert tone")
#             self.last_alert_time = current_time
#             return [aframe] * len(frames)
#         logger.debug(f"Fatigued state: {self.fatigued}, Last alert: {current_time - self.last_alert_time}")
#         return frames

# # ---------------- Streamlit UI ----------------
# st.set_page_config(page_title="Drowsiness Detection", page_icon="😴", layout="centered")
# st.title("😴 Real-time Drowsiness Detection")
# st.markdown("Use the sidebar to configure detection rules.")

# st.sidebar.header("⚙️ Settings")
# conditions = st.sidebar.multiselect(
#     "Select triggers for fatigue:",
#     ["Left Eye Closed", "Right Eye Closed", "Mouth Open", "Use PERCLOS"],
#     default=["Left Eye Closed", "Right Eye Closed", "Mouth Open", "Use PERCLOS"]
# )
# perclos_threshold = st.sidebar.slider("PERCLOS Threshold", 0.1, 1.0, 0.4)

# if "start" not in st.session_state:
#     st.session_state.start = False

# if st.sidebar.button("▶️ Start Detection"):
#     st.session_state.start = True

# # ---------------- MediaPipe Setup ----------------
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.3,
#     min_tracking_confidence=0.5
# )

# LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
# MOUTH_POINTS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17,
#                 84, 181, 91, 146, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324,
#                 318, 402, 317, 14, 87, 178, 88, 95]

# # ---------------- Video Processor ----------------
# class VideoProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.audio_processor = None
#         self.list_B = np.ones(int(VIDEO_FPS * 3))
#         self.list_Y = np.zeros(int(VIDEO_FPS * 10))
#         self.list_blink = np.zeros(int(VIDEO_FPS * 10))
#         self.list_yawn = np.zeros(int(VIDEO_FPS * 30))
#         self.frame_count = 0
#         self.process_every_n_frames = 3
#         self.left_eye_counter = 0
#         self.right_eye_counter = 0
#         self.last_fatigue_time = 0
#         self.list_Y1 = np.ones(int(VIDEO_FPS * 2))

#     async def recv_queued(self, frames: list[av.VideoFrame]) -> list[av.VideoFrame]:
#         if not frames:
#             return frames
        
#         frame = frames[-1]
#         self.frame_count += 1
#         if self.frame_count % self.process_every_n_frames != 0:
#             return frames

#         img = frame.to_ndarray(format="bgr24")
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         h, w, _ = rgb.shape
#         status_text = "Alert"
#         current_time = asyncio.get_event_loop().time()

#         try:
#             results = face_mesh.process(rgb)
#             logger.debug(f"Face detected: {bool(results.multi_face_landmarks)}")
#             if results.multi_face_landmarks:
#                 for lmset in results.multi_face_landmarks:
#                     lm = lmset.landmark
#                     left_eye, left_coords = extract_roi(rgb, lm, LEFT_EYE_POINTS, img_width=w, img_height=h)
#                     right_eye, right_coords = extract_roi(rgb, lm, RIGHT_EYE_POINTS, img_width=w, img_height=h)
#                     mouth, mouth_coords = extract_roi(rgb, lm, MOUTH_POINTS, is_mouth=True, img_width=w, img_height=h)

#                     flag_B = False
#                     flag_Y = False

#                     # Predictions
#                     if left_eye is not None:
#                         pred_idx, _ = predict_drowsiness(left_eye, True)
#                         if pred_idx == 0:
#                             flag_B = True
#                             self.left_eye_counter += 1
#                             if self.left_eye_counter >= EYE_AR_CONSEC_FRAMES:
#                                 logger.info("Left eye blink detected")
#                                 self.left_eye_counter = 0
#                         else:
#                             self.left_eye_counter = 0
#                     if right_eye is not None:
#                         pred_idx, _ = predict_drowsiness(right_eye, True)
#                         if pred_idx == 0:
#                             flag_B = True
#                             self.right_eye_counter += 1
#                             if self.right_eye_counter >= EYE_AR_CONSEC_FRAMES:
#                                 logger.info("Right eye blink detected")
#                                 self.right_eye_counter = 0
#                         else:
#                             self.right_eye_counter = 0
#                     if mouth is not None:
#                         pred_idx, _ = predict_drowsiness(mouth, False)
#                         if pred_idx == 3:
#                             flag_Y = True

#                     # Update metrics lists
#                     self.list_B = np.append(self.list_B, 0 if flag_B else 1)
#                     self.list_B = np.delete(self.list_B, 0)
#                     self.list_Y = np.append(self.list_Y, 1 if flag_Y else 0)
#                     self.list_Y = np.delete(self.list_Y, 0)
#                     self.list_blink = np.append(self.list_blink, 1 if len(self.list_B) >= 2 and self.list_B[-2] == 1 and self.list_B[-1] == 0 else 0)
#                     self.list_blink = np.delete(self.list_blink, 0)
#                     if len(self.list_Y) >= len(self.list_Y1) and (self.list_Y[-len(self.list_Y1):] == self.list_Y1).all():
#                         logger.info("-------- Yawn --------")
#                         self.list_Y = np.zeros(int(VIDEO_FPS * 10))
#                         self.list_yawn = np.append(self.list_yawn, 1)
#                     else:
#                         self.list_yawn = np.append(self.list_yawn, 0)
#                     self.list_yawn = np.delete(self.list_yawn, 0)

#                     perclos = 1 - np.average(self.list_B)
#                     perblink = np.average(self.list_blink)
#                     peryawn = np.average(self.list_yawn)
#                     fatigue_status = "Fatigued" if (perclos > 0.4 or perblink < 2.5 / (10 * VIDEO_FPS) or peryawn > 3 / (30 * VIDEO_FPS)) else "Alert"

#                     # Draw ROI boxes
#                     if left_coords:
#                         cv2.rectangle(img, (left_coords[0], left_coords[1]), (left_coords[2], left_coords[3]), (255, 0, 0), 2)
#                     if right_coords:
#                         cv2.rectangle(img, (right_coords[0], right_coords[1]), (right_coords[2], right_coords[3]), (255, 0, 0), 2)
#                     if mouth_coords:
#                         cv2.rectangle(img, (mouth_coords[0], mouth_coords[1]), (mouth_coords[2], mouth_coords[3]), (0, 0, 255), 2)

#                     # Overlay text
#                     left_eye_pred = predict_drowsiness(left_eye, True)[0] if left_eye is not None else None
#                     right_eye_pred = predict_drowsiness(right_eye, True)[0] if right_eye is not None else None
#                     mouth_pred = predict_drowsiness(mouth, False)[0] if mouth is not None else None

#                     if (current_time - self.last_fatigue_time) > FATIGUE_COOLDOWN or fatigue_status == "Alert":
#                         status_text = fatigue_status
#                         self.last_fatigue_time = current_time if fatigue_status == "Fatigued" else self.last_fatigue_time
#                     else:
#                         status_text = "Alert"

#                     for idx in LEFT_EYE_POINTS + RIGHT_EYE_POINTS + MOUTH_POINTS:
#                         x = int(lm[idx].x * w)
#                         y = int(lm[idx].y * h)
#                         cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
#         except Exception as e:
#             logger.error(f"Processing error: {e}")
#             status_text = "Error"

#         if self.audio_processor:
#             self.audio_processor.fatigued = (status_text == "Fatigued")

#         cv2.putText(img, f"Status: {status_text}", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (0, 0, 255) if status_text == "Fatigued" else (0, 255, 0), 2)

#         processed_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
#         return [processed_frame] * len(frames)

# # ---------------- WebRTC Config ----------------
# RTC_CFG = RTCConfiguration({
#     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
# })

# # ---------------- Run ----------------
# if st.session_state.start:
#     try:
#         ctx = webrtc_streamer(
#             key="drowsiness-app",
#             rtc_configuration=RTC_CFG,
#             media_stream_constraints={"video": True, "audio": True},
#             video_processor_factory=VideoProcessor,
#             audio_processor_factory=AlarmAudioProcessor,
#             async_processing=True,
#         )
#         if ctx.video_processor and ctx.audio_processor:
#             ctx.video_processor.audio_processor = ctx.audio_processor
#     except Exception as e:
#         st.error(f"WebRTC error: {e}")
#         logger.error(f"WebRTC error: {e}")
# else:
#     st.info("👆 Use the sidebar to configure rules and press Start to launch webcam + mic.")



import av
import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tflite
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, AudioProcessorBase, RTCConfiguration
import logging
import asyncio

# ---------------- CONFIG ----------------
IMG_WIDTH, IMG_HEIGHT = 224, 224
MODEL_PATH = "model/drowsiness_model.tflite"
MOUTH_ROI_SCALE = 1.5
PADDING = 10
VIDEO_FPS = 20
EYE_AR_CONSEC_FRAMES = 3
FATIGUE_COOLDOWN = 5.0

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ---------------- Load TFLite ----------------
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("TFLite model loaded successfully")
except Exception as e:
    st.error(f"Failed to load TFLite model: {e}")
    logger.error(f"Model loading error: {e}")
    st.stop()

# ---------------- Helper Functions ----------------
def preprocess_image(img):
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def extract_roi(image, landmarks, points, is_mouth=False, img_width=None, img_height=None):
    x_coords = [landmarks[i].x * img_width for i in points]
    y_coords = [landmarks[i].y * img_height for i in points]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    if is_mouth:
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, x_min - int(width * (MOUTH_ROI_SCALE - 1) / 2))
        x_max = min(image.shape[1], x_max + int(width * (MOUTH_ROI_SCALE - 1) / 2))
        y_min = max(0, y_min - int(height * (MOUTH_ROI_SCALE - 1) / 2))
        y_max = min(image.shape[0], y_max + int(height * (MOUTH_ROI_SCALE - 1) / 2))
    else:
        x_min = max(0, x_min - PADDING)
        x_max = min(image.shape[1], x_max + PADDING)
        y_min = max(0, y_min - PADDING)
        y_max = min(image.shape[0], y_max + PADDING)

    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        logger.warning("Empty ROI detected")
        return None, None
    return roi, (x_min, y_min, x_max, y_max)

def predict_drowsiness(roi, is_eye=True):
    if roi is None:
        return None
    try:
        input_data = preprocess_image(roi)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        if is_eye:
            return np.argmax(prediction[:2]), prediction[:2]  # 0=Closed, 1=Open
        else:
            return np.argmax(prediction[2:]) + 2, prediction[2:]  # 2=NoYawn, 3=Yawn
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

# ---------------- Audio Alarm ----------------
class AlarmAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.fatigued = False
        self.last_alert_time = 0

    async def recv_queued(self, frames: list[av.AudioFrame]) -> list[av.AudioFrame]:
        if not frames:
            return frames
        
        current_time = asyncio.get_event_loop().time()
        if self.fatigued and (current_time - self.last_alert_time) > 1.0:
            sr = 48000
            samples = np.arange(sr)  # 1-second tone
            tone = (np.sin(2 * np.pi * 440 * samples / sr) * 0.7).astype(np.float32)
            aframe = av.AudioFrame.from_ndarray(tone, layout="mono")
            aframe.sample_rate = sr
            logger.debug("Playing fatigue alert tone")
            self.last_alert_time = current_time
            return [aframe] * len(frames)
        return frames

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Drowsiness Detection", page_icon="😴", layout="centered")
st.title("😴 Real-time Drowsiness Detection")
st.markdown("Use the sidebar to configure detection rules.")

st.sidebar.header("⚙️ Settings")
conditions = st.sidebar.multiselect(
    "Select triggers for fatigue:",
    ["Left Eye Closed", "Right Eye Closed", "Mouth Open", "Use PERCLOS"],
    default=["Left Eye Closed", "Right Eye Closed", "Mouth Open", "Use PERCLOS"]
)
perclos_threshold = st.sidebar.slider("PERCLOS Threshold", 0.1, 1.0, 0.4)

if "start" not in st.session_state:
    st.session_state.start = False

if st.sidebar.button("▶️ Start Detection"):
    st.session_state.start = True

# ---------------- MediaPipe Setup ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
MOUTH_POINTS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17,
                84, 181, 91, 146, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324,
                318, 402, 317, 14, 87, 178, 88, 95]

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.audio_processor = None
        self.list_B = np.ones(int(VIDEO_FPS * 3))
        self.list_Y = np.zeros(int(VIDEO_FPS * 10))
        self.list_blink = np.zeros(int(VIDEO_FPS * 10))
        self.list_yawn = np.zeros(int(VIDEO_FPS * 30))
        self.frame_count = 0
        self.process_every_n_frames = 3
        self.left_eye_counter = 0
        self.right_eye_counter = 0
        self.last_fatigue_time = 0
        self.list_Y1 = np.ones(int(VIDEO_FPS * 2))

    async def recv_queued(self, frames: list[av.VideoFrame]) -> list[av.VideoFrame]:
        if not frames:
            return frames
        
        frame = frames[-1]
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return frames

        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        status_text = "No Face Detected"
        current_time = asyncio.get_event_loop().time()

        try:
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                status_text = "Alert"  # Reset status to standard alert
                for lmset in results.multi_face_landmarks:
                    lm = lmset.landmark
                    left_eye, left_coords = extract_roi(rgb, lm, LEFT_EYE_POINTS, img_width=w, img_height=h)
                    right_eye, right_coords = extract_roi(rgb, lm, RIGHT_EYE_POINTS, img_width=w, img_height=h)
                    mouth, mouth_coords = extract_roi(rgb, lm, MOUTH_POINTS, is_mouth=True, img_width=w, img_height=h)

                    flag_B = False
                    flag_Y = False

                    # Predictions
                    if left_eye is not None:
                        pred_idx, _ = predict_drowsiness(left_eye, True)
                        if pred_idx == 0:
                            flag_B = True
                            self.left_eye_counter += 1
                            if self.left_eye_counter >= EYE_AR_CONSEC_FRAMES:
                                self.left_eye_counter = 0
                        else:
                            self.left_eye_counter = 0
                    if right_eye is not None:
                        pred_idx, _ = predict_drowsiness(right_eye, True)
                        if pred_idx == 0:
                            flag_B = True
                            self.right_eye_counter += 1
                            if self.right_eye_counter >= EYE_AR_CONSEC_FRAMES:
                                self.right_eye_counter = 0
                        else:
                            self.right_eye_counter = 0
                    if mouth is not None:
                        pred_idx, _ = predict_drowsiness(mouth, False)
                        if pred_idx == 3:
                            flag_Y = True

                    # Update metrics lists
                    self.list_B = np.append(self.list_B, 0 if flag_B else 1)
                    self.list_B = np.delete(self.list_B, 0)
                    self.list_Y = np.append(self.list_Y, 1 if flag_Y else 0)
                    self.list_Y = np.delete(self.list_Y, 0)
                    self.list_blink = np.append(self.list_blink, 1 if len(self.list_B) >= 2 and self.list_B[-2] == 1 and self.list_B[-1] == 0 else 0)
                    self.list_blink = np.delete(self.list_blink, 0)
                    if len(self.list_Y) >= len(self.list_Y1) and (self.list_Y[-len(self.list_Y1):] == self.list_Y1).all():
                        self.list_Y = np.zeros(int(VIDEO_FPS * 10))
                        self.list_yawn = np.append(self.list_yawn, 1)
                    else:
                        self.list_yawn = np.append(self.list_yawn, 0)
                    self.list_yawn = np.delete(self.list_yawn, 0)

                    perclos = 1 - np.average(self.list_B)
                    perblink = np.average(self.list_blink)
                    peryawn = np.average(self.list_yawn)
                    fatigue_status = "Fatigued" if (perclos > 0.4 or perblink < 2.5 / (10 * VIDEO_FPS) or peryawn > 3 / (30 * VIDEO_FPS)) else "Alert"

                    # Draw ROI boxes
                    if left_coords:
                        cv2.rectangle(img, (left_coords[0], left_coords[1]), (left_coords[2], left_coords[3]), (255, 0, 0), 2)
                    if right_coords:
                        cv2.rectangle(img, (right_coords[0], right_coords[1]), (right_coords[2], right_coords[3]), (255, 0, 0), 2)
                    if mouth_coords:
                        cv2.rectangle(img, (mouth_coords[0], mouth_coords[1]), (mouth_coords[2], mouth_coords[3]), (0, 0, 255), 2)

                    # Apply correct status
                    if (current_time - self.last_fatigue_time) > FATIGUE_COOLDOWN or fatigue_status == "Alert":
                        status_text = fatigue_status
                        self.last_fatigue_time = current_time if fatigue_status == "Fatigued" else self.last_fatigue_time

        except Exception as e:
            logger.error(f"Processing error: {e}")
            status_text = "Error"

        if self.audio_processor:
            self.audio_processor.fatigued = (status_text == "Fatigued")

        cv2.putText(img, f"Status: {status_text}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if status_text == "Fatigued" else (0, 255, 0), 2)

        processed_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        return [processed_frame] * len(frames)

# ---------------- WebRTC Config ----------------
RTC_CFG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ---------------- Run ----------------
if st.session_state.start:
    try:
        ctx = webrtc_streamer(
            key="drowsiness-app",
            rtc_configuration=RTC_CFG,
            media_stream_constraints={"video": True, "audio": True},
            video_processor_factory=VideoProcessor,
            audio_processor_factory=AlarmAudioProcessor,
            async_processing=True,
        )
        if ctx.video_processor and ctx.audio_processor:
            ctx.video_processor.audio_processor = ctx.audio_processor
    except Exception as e:
        st.error(f"WebRTC error: {e}")
        logger.error(f"WebRTC error: {e}")
else:
    st.info("👆 Use the sidebar to configure rules and press Start to launch webcam + mic.")