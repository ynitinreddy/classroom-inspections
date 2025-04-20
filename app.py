import base64
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
import os, openai, streamlit as st
import torch
from docx import Document
from docx.shared import Inches
import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv(dotenv_path=r"C:\Users\yniti\Downloads\classroom_inspector_api_key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

torch.classes.__path__ = []

# --- Page Config ---
st.set_page_config(page_title="AI Classroom Inspector", layout="wide")

# --- ASU Styling Refined ---
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
        background-color: #ffffff;
        color: #000000;
    }
    .stButton>button {
        color: white;
        background-color: #8C1D40;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 16px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #FFC627;
        color: black;
    }
    .stSelectbox label, .stCheckbox label, .stTextArea label, .stSubheader, .stCaption, .stMarkdown, .stTextInput label {
        color: #8C1D40 !important;
    }
    .stTextInput>div>input {
        background-color: #fff8dc;
        border: 1px solid #8C1D40;
        color: black;
    }
    .stTextArea textarea {
        background-color: #fff8dc;
        border: 1px solid #8C1D40;
        color: black;
    }
    .about-photo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 50%;
        width: 150px;
        height: 150px;
        object-fit: cover;
    }
    .about-container {
        text-align: center;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------
# 1.  Encapsulate model loading in a cached function
# -----------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

# Helper to load images

def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Sidebar About
img_b64 = get_image_base64("musk-photo-1.jpg")
with st.sidebar.expander("📄 About This Project"):
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{img_b64}" style="width:150px; border-radius: 50%;" />
            <div style="color:#8C1D40; font-size: 16px; margin-top: 8px;"><strong>Nitin Reddy Yarava</strong></div>
            <p style="font-size: 16px;">This project was developed as part of an initiative to automate and improve classroom inspections at ASU. It uses a hybrid approach of YOLOv8 for object detection and GPT-4 Vision for smart visual reasoning. Built by a CS student passionate about ML-driven operations and automation.</p>
        </div>
        """, unsafe_allow_html=True)

# --- UI: Steps ---
st.title("AI-Powered Classroom Inspection - ASU Edition")
st.markdown("Welcome! Upload classroom images to automatically detect issues and generate an inspection report.")

# Step 1: Image Upload
st.subheader("Step 1: Upload Classroom Images")
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader_0"

uploaded_files = st.file_uploader(
    "Upload classroom images (from different angles if possible)",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"],
    key=st.session_state.uploader_key
)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

if st.session_state.uploaded_files and st.button("🗑️ Clear Uploaded Images"):
    st.session_state.uploaded_files = []
    idx = int(st.session_state.uploader_key.split("_")[1]) + 1
    st.session_state.uploader_key = f"uploader_{idx}"
    st.rerun()

# Step 1.5: Class Number and Inspector
st.subheader("Step 1.5: Enter Class Number and Inspector")
class_number = st.text_input(
    "Enter the classroom number (e.g., 'DH 101'):",
    value=""
)
inspector_name = st.selectbox(
    "Select Inspector:",
    ["Nitin", "Jose", "Priyam", "Tanvi", "Others"],
    index=0
)

# Step 2: Model Selection
st.subheader("Step 2: Choose Model & Options")
model_choice = st.selectbox(
    "Select LLM model type:",
    ["Best (faster, lower cost)", "Basic (fastest, cheapest)", "Expert (most advanced reasoning)"],
    index=0
)
enable_yolo = st.checkbox("Detect and highlight unusual objects?", value=False)

model_map = {
    "Best (faster, lower cost)": ("gpt-4o", "Using best model."),
    "Basic (fastest, cheapest)": ("gpt-4o-mini", "Using smaller model."),
    "Expert (most advanced reasoning)": ("gpt-4o", "Using expert-level reasoning.")
}
selected_model, model_comment = model_map[model_choice]

if enable_yolo:
    yolo_model = load_yolo()
    ANOMALY_CLASSES = {0: "person", 1: "bicycle", /* ... */ 79: "toothbrush"}

# --- DOCX Generation Function ---
def generate_docx_report(report_text, original_images, anomaly_images=None, class_number=None, inspector_name=None):
    doc = Document()
    # Title
    doc.add_heading('Classroom Inspection Report', 0)
    # Class and Inspector
    if class_number:
        doc.add_paragraph(f"Class Number: {class_number}")
    else:
        doc.add_paragraph("Class Number: __________________")
    if inspector_name:
        doc.add_paragraph(f"Inspector: {inspector_name}")
    else:
        doc.add_paragraph("Inspector: __________________")
    # Date
    doc.add_paragraph(f"Date: {datetime.date.today().strftime('%B %d, %Y')}")
    doc.add_paragraph("")
    # Summary
    doc.add_heading('1. Inspection Summary', level=1)
    for line in report_text.strip().split('\n'):
        if line.strip():
            doc.add_paragraph(line.strip(), style='List Bullet')
    # Images
    doc.add_heading('2. Uploaded Classroom Images', level=1)
    for i, img_file in enumerate(original_images):
        img = Image.open(img_file)
        buf = io.BytesIO(); img.save(buf, format='JPEG'); buf.seek(0)
        doc.add_paragraph(f"Original Image {i+1}")
        doc.add_picture(buf, width=Inches(5))
        doc.add_paragraph("")
    # Anomalies
    if anomaly_images:
        doc.add_heading('3. YOLO Anomaly Detections', level=1)
        for i, img in enumerate(anomaly_images):
            buf = io.BytesIO(); img.save(buf, format='JPEG'); buf.seek(0)
            doc.add_paragraph(f"Anomaly Image {i+1}")
            doc.add_picture(buf, width=Inches(5))
            doc.add_paragraph("")
    # Filename
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    inspector_part = inspector_name.replace(" ", "_") if inspector_name else "Unknown"
    class_part = class_number.replace(" ", "_") if class_number else "Unknown"
    file_name = f"{today_str}_{inspector_part}_{class_part}_report.docx"
    # Save buffer & disk
    out_io = io.BytesIO(); doc.save(out_io); out_io.seek(0)
    os.makedirs("temp_reports", exist_ok=True)
    local_path = os.path.join("temp_reports", file_name)
    with open(local_path, "wb") as f: doc.save(f)
    return out_io, file_name, local_path

# --- Prompt Section ---
def call_gpt_hybrid(images, prompt, model, anomaly_data=None):
    # [unchanged hybrid call logic]
    ...

# --- Run Inspection ---
if st.button("🔍 Run Inspection", use_container_width=True):
    # validation
    if not st.session_state.uploaded_files:
        st.error("Please upload at least 1 image.")
    else:
        # [display images, detect anomalies...]
        report = call_gpt_hybrid(st.session_state.uploaded_files, prompt, selected_model, anomalies)
        st.subheader("Inspection Report")
        st.markdown(report)
        # Generate DOCX
        output_io, generated_file_name, local_file_path = generate_docx_report(
            report,
            st.session_state.uploaded_files,
            anomaly_images if enable_yolo else None,
            class_number,
            inspector_name
        )
        # Download button
        st.download_button(
            label="📄 Download Full Report (.docx)",
            data=output_io,
            file_name=generated_file_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        st.info(f"Report saved locally at {local_file_path}. After reviewing, upload to Drive page.")

# --- Footer ---
st.markdown("---")
st.caption("Built by Nitin, a CS student at ASU ✌️")
