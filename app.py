import base64
import datetime
import io
import os

import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from docx import Document
from docx.shared import Inches
from ultralytics import YOLO
import torch
import openai

# -----------------------------------------------------------
# 0️⃣  Environment & API Keys
# -----------------------------------------------------------
load_dotenv(dotenv_path=r"C:\Users\yniti\Downloads\classroom_inspector_api_key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------------------------------------
# 1️⃣  Streamlit Page Config & Global Styling
# -----------------------------------------------------------
st.set_page_config(page_title="AI Classroom Inspector", layout="wide")

st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding: 2rem;
        background-color: #ffffff;
        color: #000000;
    }
    .stButton>button {
        color: white;
        background-color: #8C1D40;            /* ASU maroon */
        border: none;
        padding: 0.5rem 1rem;
        font-size: 16px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #FFC627;             /* ASU gold */
        color: black;
    }
    .stSelectbox label, .stCheckbox label, .stTextArea label,
    .stSubheader, .stCaption, .stMarkdown, .stTextInput label {
        color: #8C1D40 !important;
    }
    .stTextInput>div>input, .stTextArea textarea {
        background-color: #fff8dc;             /* very‑light gold */
        border: 1px solid #8C1D40;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------
# 2️⃣  Session‑State Defaults (remember previous choices)
# -----------------------------------------------------------
DEFAULT_INSPECTOR = "Nitin"
DEFAULT_MODEL     = "Basic (fastest, cheapest)"
DEFAULT_YOLO      = True

if "inspector_name" not in st.session_state:
    st.session_state.inspector_name = DEFAULT_INSPECTOR
if "model_choice" not in st.session_state:
    st.session_state.model_choice = DEFAULT_MODEL
if "enable_yolo" not in st.session_state:
    st.session_state.enable_yolo = DEFAULT_YOLO

# -----------------------------------------------------------
# 3️⃣  Utility Functions
# -----------------------------------------------------------

def get_image_base64(path: str) -> str:
    """Return base64 string of a local image for inline HTML/Markdown."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def image_to_base64(image_file) -> str:
    img = Image.open(image_file).convert("RGB")
    buff = io.BytesIO()
    img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode()


def load_yolo_cached():
    @st.cache_resource(show_spinner="Loading YOLO model …")
    def _load():
        return YOLO("300_yolov8n.pt")  # ~6 MB, CPU‑friendly

    return _load()


def call_gpt_hybrid(images, prompt, model, anomaly_data=None):
    """Call GPT‑4o/mini with mixed image blocks and optional anomaly summary."""
    blocks = [{"type": "text", "text": prompt}]

    if st.session_state.enable_yolo:
        if anomaly_data:
            summary_lines = "\n".join(f"- {k}: {v}" for k, v in anomaly_data.items())
            blocks.insert(1, {"type": "text", "text": f"These objets were detected by an object‑detection model:\n{summary_lines}"})
        else:
            blocks.insert(1, {"type": "text", "text": "No objects were detected by the model."})

    blocks += [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(img)}"},
        }
        for img in images
    ]

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": blocks}],
        max_tokens=1000,
        temperature=0.3,
        top_p=0.8,
    )
    return resp.choices[0].message.content


def generate_docx_report(
    report_text,
    original_images,
    anomaly_images=None,
    class_number: str | None = None,
    inspector_name: str | None = None,
):
    """Return (BytesIO, filename, local_path) for a full DOCX report."""

    doc = Document()

    # Header info
    doc.add_heading("Classroom Inspection Report", 0)
    doc.add_paragraph(f"Class Number: {class_number if class_number else '__________________'}")
    doc.add_paragraph(f"Inspector: {inspector_name if inspector_name else '__________________'}")
    doc.add_paragraph(f"Date: {datetime.date.today().strftime('%B %d, %Y')}")
    doc.add_paragraph("")

    # Inspection summary
    doc.add_heading("1. Inspection Summary", level=1)
    for line in report_text.strip().split("\n"):
        if line.strip():
            doc.add_paragraph(line.strip(), style="List Bullet")

    # Original images
    doc.add_heading("2. Uploaded Classroom Images", level=1)
    for i, img_file in enumerate(original_images):
        img = Image.open(img_file)
        img_io = io.BytesIO()
        img.save(img_io, format="JPEG")
        img_io.seek(0)
        doc.add_paragraph(f"Original Image {i + 1}")
        doc.add_picture(img_io, width=Inches(5))
        doc.add_paragraph("")

    # Anomaly detections
    if anomaly_images:
        doc.add_heading("3. YOLO Enhanced Detections", level=1)
        for i, img in enumerate(anomaly_images):
            img_io = io.BytesIO()
            img.save(img_io, format="JPEG")
            img_io.seek(0)
            doc.add_paragraph(f"Recognized Image {i + 1}")
            doc.add_picture(img_io, width=Inches(5))
            doc.add_paragraph("")

    # Build filename
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    inspector_part = (
        inspector_name.strip().replace(" ", "_") if inspector_name else "anonymous"
    )
    classroom_part = (
        class_number.strip().replace(" ", "_") if class_number else "unknownclass"
    )
    file_name = f"{today_str}_{inspector_part}_{classroom_part}_report.docx"

    # Save to BytesIO and local temp dir
    output_io = io.BytesIO()
    doc.save(output_io)
    output_io.seek(0)

    os.makedirs("temp_reports", exist_ok=True)
    local_path = os.path.join("temp_reports", file_name)
    with open(local_path, "wb") as fp:
        doc.save(fp)

    return output_io, file_name, local_path


# -----------------------------------------------------------
# 4️⃣  Header & Logo
# -----------------------------------------------------------
logo_b64 = get_image_base64("ASU-logo.png")
st.markdown(
    f"<img src='data:image/png;base64,{logo_b64}' width='250'/>",
    unsafe_allow_html=True,
)
st.title("AI‑Powered Classroom Inspection · ASU Edition")
st.markdown("Upload classroom images to automatically detect issues and generate an inspection report.")

# -----------------------------------------------------------
# 5️⃣  Image Uploader
# -----------------------------------------------------------

st.subheader("Step 1: Upload Classroom Images")
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader_0"

uploaded_files = st.file_uploader(
    "Upload classroom images (multiple angles preferred)",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"],
    key=st.session_state.uploader_key,
)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

if st.session_state.uploaded_files:
    if st.button("🗑️ Clear Uploaded Images"):
        st.session_state.uploaded_files = []
        new_id = int(st.session_state.uploader_key.split("_")[1]) + 1
        st.session_state.uploader_key = f"uploader_{new_id}"
        st.rerun()

# -----------------------------------------------------------
# 6️⃣  Class No. & Inspector (Combined Step 1.5)
# -----------------------------------------------------------

st.subheader("Step 1.5: Inspection Details")

col_a, col_b = st.columns(2)
with col_a:
    class_number = st.text_input(
        "Classroom Number (e.g., 'DH 101') – leave blank if unknown",
        value=st.session_state.get("class_number", ""),
        key="class_number",
    )

with col_b:
    inspector_options = ["Nitin", "Jose", "Priyam", "Tanvi", "Others"]
    inspector_name = st.selectbox(
        "Inspector Name",
        inspector_options,
        index=inspector_options.index(st.session_state.inspector_name)
        if st.session_state.inspector_name in inspector_options
        else 0,
        key="inspector_name",
    )

# If "Others", prompt for custom name
custom_name = ""
if inspector_name == "Others":
    custom_name = st.text_input(
        "Enter inspector name:",
        value=st.session_state.get("custom_inspector_name", ""),
        key="custom_inspector_name",
    )

inspector_used = (
    custom_name.strip() if inspector_name == "Others" else inspector_name
)
if inspector_used == "":
    inspector_used = "anonymous"

# -----------------------------------------------------------
# 7️⃣  Model & YOLO Options (remembered)
# -----------------------------------------------------------

st.subheader("Step 2: Choose Model & Options")
model_options = [
    "Best (faster, lower cost)",
    "Basic (fastest, cheapest)",
    "Expert (most advanced reasoning for images) – need to add",
]
model_choice = st.selectbox(
    "LLM Model:",
    model_options,
    index=model_options.index(st.session_state.model_choice)
    if st.session_state.model_choice in model_options
    else model_options.index(DEFAULT_MODEL),
    key="model_choice",
)

enable_yolo_checkbox = st.checkbox(
    "Detect and highlight unusual objects?",
    value=st.session_state.enable_yolo,
    key="enable_yolo",
)

# After widgets, capture the current selections without touching their keys
enable_yolo = st.session_state.enable_yolo  # current checkbox state

# -----------------------------------------------------------
# 8️⃣  Model Mapping & YOLO Init (lazy)
# -----------------------------------------------------------

model_map = {
    "Best (faster, lower cost)": ("gpt-4o", "Using best model."),
    "Basic (fastest, cheapest)": ("gpt-4o-mini", "Using smaller model."),
    "Expert (most advanced reasoning for images) – need to add": (
        "gpt-4o",
        "Using expert-level reasoning model (gpt-4o).",
    ),
}
selected_model, model_comment = model_map[model_choice]

if st.session_state.enable_yolo:
    yolo_model = load_yolo_cached()
    # ONLY use this if those indexes are *really* what your .pt was trained on:
    # CUSTOM_CLASSES = {
    #     0: "911 Address",
    #     1: "Bill of Rights Constitution",
    #     2: "Capacity Sign",
    #     3: "Classroom Layout",
    #     4: "Classroom Support Pocket",
    #     5: "Clock",
    #     6: "Dirty Whiteboard",
    #     7: "ERG Layout",
    #     8: "Exit Sign",
    #     9: "Flag",
    #     10: "Miscellaneous Objects",
    #     11: "No Food/Drinks Sign",
    #     12: "Recycle Bin",
    #     13: "Scrapes",
    #     14: "Stains",
    #     15: "Trash Bin",
    #     16: "University Classrooms Pocket",
    #     17: "Whiteboard",
    #     18: "Window Covering",
    # }

    CUSTOM_CLASSES = {
        0: "911 Address",
        1: "Bill of Rights & Constitution",
        2: "Bins",
        3: "Capacity Sign",
        4: "Classroom Support Pocket",
        5: "Clock",
        6: "ERG",
        7: "Exit Sign",
        8: "Flag",
        9: "No Food or Drinks Sign",
        10: "Scuffs/Scrapes",
        11: "Stains",
        12: "UCL Pocket",
        13: "Whiteboard",
    }

    def detect_objects(images):
        counts: dict[str, int] = {}
        annotated: list[Image.Image] = []

        for img_file in images:
            img = Image.open(img_file).convert("RGB")
            results = yolo_model(np.array(img), classes=list(CUSTOM_CLASSES.keys()))

            for result in results:
                if not result.boxes:
                    continue

                # Filter boxes by confidence
                high_conf_boxes = []
                for box in result.boxes:
                    if box.conf[0] >= 0.45:
                        cls = int(box.cls[0])
                        label = CUSTOM_CLASSES[cls]
                        counts[label] = counts.get(label, 0) + 1
                        high_conf_boxes.append(box)

                # Update result.boxes with only high-confidence boxes
                result.boxes = type(result.boxes)(high_conf_boxes)  # keep only confident ones

                # Plot filtered boxes
                annotated.append(Image.fromarray(result.plot(conf=True, labels=True)))

        return counts, annotated


else:
    detect_objects = None



# -----------------------------------------------------------
# 9️⃣  Prompt Engineering
# -----------------------------------------------------------

def build_default_prompt(use_yolo: bool) -> str:
    extra = " and object counts" if use_yolo else ""
    return f"""
{model_comment}

You are a classroom inspection assistant. You will be given images{extra}.
⚠️ VERY IMPORTANT: Keep each of the 12 items to one very short sentence (“No problems found.” if OK).

Use a numbered list 1–12. For each:
- Start with the heading (e.g., “Walls:”).
- If the feature is in the image, say “Present – [very brief detail]”.
- If it’s not there, say “Absent.”
- If you can’t tell, say “Cannot determine.”

1. Side Walls: Present/Absent. If present, note scuffs, holes, etc.
2. Ceiling: Present/Absent. If present, note holes, stains, etc.
3. White Board: Present/Absent. If present, note cleanliness or writing.
4. Floor: Present/Absent. If present, note trash, stains, tears.
5. Bins: Present/Absent. If present, count and type (trash/recycle).
6. Exit Sign: Present/Absent.
7. Lights: Present/Absent. If present, note any bulbs out.
8. Flag: Present/Absent.
9. “No Food/Drinks” Plaque: Present/Absent.
10. Instructor’s Desk: Present/Absent. If present, note cleanliness.
11. Clock: Present/Absent/Unsure.
12: Capacity Sign: Present/Absent/Unsure
13. UCL Pocket: Present/Absent/Unsure
14. Classroom Support Pocket: Present/Absent/Unsure
15. Additional Comments: Any unusual items or safety issues.
"""

prompt_default = build_default_prompt(st.session_state.enable_yolo)
with st.expander("⚙️ More Options: Edit Inspection Prompt"):
    prompt = st.text_area("LLM Prompt", prompt_default, height=260)

# -----------------------------------------------------------
# 🔟  Run Inspection
# -----------------------------------------------------------

center = st.columns([1, 2, 1])[1]
run_btn = center.button("🔍 Run Inspection", use_container_width=True)
status = st.empty()

if run_btn:
    if not st.session_state.uploaded_files:
        st.error("Please upload at least one image.")
        st.stop()

    # 1) Show uploaded images
    status.info("Preparing images…")
    with st.expander("📷 View uploaded images"):
        cols = st.columns(min(len(st.session_state.uploaded_files), 4))
        for i, f in enumerate(st.session_state.uploaded_files):
            with cols[i % 4]:
                st.image(Image.open(f), caption=f"Image {i + 1}", use_container_width=True)
    
    st.write("📝 Model reports these classes:", load_yolo_cached().model.names)


    # 2) YOLO object detection (optional)
    detections = {}
    annotated_imgs = []
    if st.session_state.enable_yolo and detect_objects:
        status.info("Detecting objects with YOLO…")
        detections, annotated_imgs = detect_objects(st.session_state.uploaded_files)
        st.write("**Detected objects:**", detections or "No objects detected.")

        if annotated_imgs:
            with st.expander("📦 YOLO Detections"):
                cols = st.columns(min(len(annotated_imgs), 4))
                for i, im in enumerate(annotated_imgs):
                    with cols[i % 4]:
                        st.image(im, caption=f"Detection {i + 1}", use_container_width=True)

        # 3) Run GPT inspection over images + detection summary
        status.info("Analyzing classroom with AI Vision…")
        report = call_gpt_hybrid(
            st.session_state.uploaded_files,
            prompt,
            selected_model,
            anomaly_data=detections if st.session_state.enable_yolo else None,
        )


    # 4) Display the 13‑point inspection report
    st.subheader("📝 Inspection Report")
    st.markdown(report)

    # 5) Generate and offer download of DOCX report
    bio, file_name, local_file_path = generate_docx_report(
        report,
        st.session_state.uploaded_files,
        anomaly_images=annotated_imgs,
        class_number=class_number,
        inspector_name=inspector_used,
    )
    st.download_button("📄 Download DOCX Report", data=bio, file_name=file_name)

    status.success("All done! 🎉")


# -----------------------------------------------------------
# 1️⃣1️⃣  Sidebar – About
# -----------------------------------------------------------

with st.sidebar.expander("📄 About This Project"):
    avatar_b64 = get_image_base64("musk-photo-1.jpg")
    st.markdown(
        f"""
        <div style='text-align:center;'>
            <img src='data:image/jpeg;base64,{avatar_b64}' style='width:150px;border-radius:50%;'/>
            <div style='color:#8C1D40;font-size:16px;margin-top:8px;'><strong>Nitin Reddy Yarava</strong></div>
            <p style='font-size:16px;'>
                This project automates ASU classroom inspections using YOLOv8 for object detection and GPT‑4 Vision for reasoning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------
# 1️⃣2️⃣  Footer
# -----------------------------------------------------------

st.markdown("---")
st.caption("Built by Nitin, a CS student at ASU ✌️")
