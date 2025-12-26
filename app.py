import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Fire & Smoke Detection", layout="wide")

st.title("🔥🚬 Fire & Smoke Detection (YOLO)")
st.caption("Upload an image or video and run inference with your trained model.")

@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)

# ---- Sidebar ----
st.sidebar.header("Settings")

model_path = r"C:\Users\Eman Yaser\Documents\AiTronix\Computer Vision\model deployment\best.pt"

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("IoU (NMS) threshold", 0.10, 0.90, 0.45, 0.05)

device = "cpu"
run_btn = st.sidebar.button("Run Inference")

# ---- Load model ----
try:
    model = load_model(model_path)
    names = model.names  # dict: {id: name}
    class_labels = [names[i] for i in sorted(names.keys())]
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

selected_classes = st.sidebar.multiselect(
    "Classes to show",
    options=class_labels,
    default=class_labels
)

# map selected class names -> ids
selected_class_ids = [i for i, n in names.items() if n in selected_classes]

# ---- Input type ----
tab1, tab2 = st.tabs(["Image", "Video"])

# =======================
# Image inference
# =======================
with tab1:
    img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(img, use_container_width=True)

        if run_btn:
            # YOLO expects numpy BGR or RGB; Ultralytics handles PIL/numpy
            results = model.predict(
                source=np.array(img),
                conf=conf,
                iou=iou,
                classes=selected_class_ids if selected_class_ids else None,
                device=device
            )

            annotated = results[0].plot()  # numpy BGR
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("Detections")
                st.image(annotated_rgb, use_container_width=True)

                # download annotated image
                out_pil = Image.fromarray(annotated_rgb)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                out_pil.save(tmp.name, format="PNG")

                with open(tmp.name, "rb") as f:
                    st.download_button(
                        label="Download result image",
                        data=f,
                        file_name="result.png",
                        mime="image/png"
                    )

# =======================
# Video inference
# =======================
with tab2:
    vid_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
    if vid_file:
        # save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        tfile.close()

        st.video(tfile.name)

        if run_btn:
            st.info("Running video inference... this may take a bit.")

            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_path = os.path.join(tempfile.gettempdir(), "result.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            frame_placeholder = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    classes=selected_class_ids if selected_class_ids else None,
                    device=device,
                    verbose=False
                )

                annotated = results[0].plot()  # BGR
                writer.write(annotated)

                # show preview
                preview = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(preview, use_container_width=True)

            cap.release()
            writer.release()

            st.success("Done ✅")
            st.video(out_path)

            with open(out_path, "rb") as f:
                st.download_button(
                    label="Download result video",
                    data=f,
                    file_name="result.mp4",
                    mime="video/mp4"
                )
