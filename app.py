import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from PIL import Image
import os
try:
    from mediapipe.python.solutions.face_mesh import FaceMesh
except ImportError:
    import mediapipe as mp
    FaceMesh = mp.solutions.face_mesh.FaceMesh
# import mediapipe as mp
# mp_face_mesh = mp.solutions.face_mesh

# ---------------------------
# LOAD GLASSES
# ---------------------------
GLASSES_FOLDER = "frames"

def load_glasses_image(path):
    img = Image.open(path).convert("RGBA")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

glasses_files = [
    os.path.join(GLASSES_FOLDER, f)
    for f in os.listdir(GLASSES_FOLDER)
    if f.lower().endswith(".png")
]

glasses_files.sort()
glasses_images = [load_glasses_image(f) for f in glasses_files]

# ---------------------------
# FACE SHAPE DETECTION
# ---------------------------
def detect_face_shape(landmarks, w, h):
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
    jaw_l, jaw_r = pts[234], pts[454]
    cheek_l, cheek_r = pts[123], pts[352]
    fore_l, fore_r = pts[10], pts[338]
    chin, top = pts[152], pts[10]

    jaw = np.linalg.norm(jaw_l - jaw_r)
    cheek = np.linalg.norm(cheek_l - cheek_r)
    fore = np.linalg.norm(fore_l - fore_r)
    height = np.linalg.norm(chin - top)

    if abs(cheek - height) < 50:
        return "Round"
    if abs(jaw - cheek) < 40 and abs(cheek - fore) < 40:
        return "Square"
    if cheek > fore and cheek > jaw and height > cheek:
        return "Oval"
    if fore > cheek > jaw:
        return "Heart"
    if height > cheek * 1.3:
        return "Long"

    return "Oval"

def recommend_glasses(shape):
    rec = {
        "Oval": "Almost any frame style works well — especially rectangular and square frames.",
        "Round": "Rectangular and square frames are recommended to add structure.",
        "Square": "Round and oval frames soften strong jawlines and balance the face.",
        "Heart": "Aviator, round, and bottom-heavy frames help balance wider foreheads.",
        "Long": "Tall frames, oversized styles, or round frames add visual balance.",
    }
    return rec.get(shape, "Most standard frame shapes suit this face type.")

# ---------------------------
# ALPHA OVERLAY
# ---------------------------
def overlay_alpha(frame, overlay, x, y):
    h, w = overlay.shape[:2]
    if x >= frame.shape[1] or y >= frame.shape[0]:
        return frame
    x1, x2 = max(0, x), min(frame.shape[1], x + w)
    y1, y2 = max(0, y), min(frame.shape[0], y + h)
    overlay_crop = overlay[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha = overlay_crop[:, :, 3:4] / 255.0
    frame[y1:y2, x1:x2] = (1 - alpha) * frame[y1:y2, x1:x2] + alpha * overlay_crop[:, :, :3]
    return frame

# ---------------------------
# VIDEO PROCESSOR
# ---------------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.selected_glasses = 0
        self.face_shape = "Detecting..."
        self.recommendation = ""
        # self.face_mesh = mp_face_mesh.FaceMesh(
        #     max_num_faces=1,
        #     refine_landmarks=True,
        #     min_detection_confidence=0.5,
        #     min_tracking_confidence=0.5
        # )
        self.face_mesh = FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        h, w = frm.shape[:2]
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            self.face_shape = detect_face_shape(lm, w, h)
            self.recommendation = recommend_glasses(self.face_shape)

            left_eye = lm[33]
            right_eye = lm[263]
            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)

            glasses_w = int(np.linalg.norm([lx - rx, ly - ry]) * 1.8)
            glasses_h = int(glasses_w * 0.40)
            selected_glasses = glasses_images[self.selected_glasses]
            glasses_resized = cv2.resize(selected_glasses, (glasses_w, glasses_h))

            dx = rx - lx
            dy = ry - ly
            angle = -np.degrees(np.arctan2(dy, dx))
            M = cv2.getRotationMatrix2D((glasses_w // 2, glasses_h // 2), angle, 1.0)
            glasses_rotated = cv2.warpAffine(
                glasses_resized,
                M,
                (glasses_w, glasses_h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )

            gx = int((lx + rx) / 2 - glasses_w / 2)
            gy = int((ly + ry) / 2 - glasses_h * 0.36)
            frm = overlay_alpha(frm, glasses_rotated, gx, gy)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("👓 AR Glasses Try-On App")
st.sidebar.title("Choose Glasses")

if "selected_glasses" not in st.session_state:
    st.session_state.selected_glasses = 0

# Sidebar: prikaži sve okvire jedan ispod drugog sa stiklicom za selektovani
for i, (img, file_path) in enumerate(zip(glasses_images, glasses_files)):
    filename = os.path.basename(file_path)
    st.sidebar.image(img[:, :, :3], width=120, caption=filename)
    
    selected = st.session_state.selected_glasses == i
    # Kompaktni raspored dugmeta i stiklice
    col1, col2 = st.sidebar.columns([1, 2])
    with col1:
        if selected:
            st.markdown("✔️")
    with col2:
        # Dugme sa imenom fajla bez ekstenzije
        button_label = os.path.splitext(filename)[0]
        if st.button(f"{button_label}", key=f"btn_{i}"):
            st.session_state.selected_glasses = i

# Camera
ctx = webrtc_streamer(
    key="glasses-demo",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# Face shape display
st.subheader("📌 Detected Face Shape & Recommendation")
placeholder_shape = st.empty()
placeholder_rec = st.empty()

if ctx.video_processor:
    ctx.video_processor.selected_glasses = st.session_state.selected_glasses
    placeholder_shape.write(f"**Face Shape:** {ctx.video_processor.face_shape}")
    placeholder_rec.write(f"**Recommended Glasses:** {ctx.video_processor.recommendation}")
