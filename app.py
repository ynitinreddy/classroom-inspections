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
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def image_to_base64(image_file) -> str:
    img = Image.open(image_file).convert("RGB")
    buff = io.BytesIO()
    img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode()


def load_yolo_cached():
    @st.cache_resource(show_spinner="Loading YOLO model …")
    def _load():
        return YOLO("classroom_yolo.pt")
    return _load()


def call_gpt_hybrid(images, prompt, model, detection_data=None):
    blocks = [{"type": "text", "text": prompt}]

    if st.session_state.enable_yolo:
        if detection_data:
            summary_lines = "\n".join(f"- {k}: {v}" for k, v in detection_data.items())
            blocks.insert(1, {"type": "text", "text": f"Detected objects by YOLO:\n{summary_lines}"})
        else:
            blocks.insert(1, {"type": "text", "text": "No objects detected by YOLO."})

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
    annotated_images=None,
    class_number: str | None = None,
    inspector_name: str | None = None,
):
    doc = Document()
    doc.add_heading("Classroom Inspection Report", 0)
    doc.add_paragraph(f"Class Number: {class_number or '__________________'}")
    doc.add_paragraph(f"Inspector: {inspector_name or '__________________'}")
    doc.add_paragraph(f"Date: {datetime.date.today().strftime('%B %d, %Y')}")
    doc.add_paragraph("")

    doc.add_heading("1. Inspection Summary", level=1)
    for line in report_text.strip().split("\n"):
        if line.strip():
            doc.add_paragraph(line.strip(), style="List Bullet")

    doc.add_heading("2. Uploaded Images", level=1)
    for i, fpath in enumerate(original_images):
        img = Image.open(fpath)
        img_io = io.BytesIO()
        img.save(img_io, format="JPEG")
        img_io.seek(0)
        doc.add_paragraph(f"Original Image {i+1}")
        doc.add_picture(img_io, width=Inches(5))
        doc.add_paragraph("")

    if annotated_images:
        doc.add_heading("3. YOLO Detections", level=1)
        for i, img in enumerate(annotated_images):
            img_io = io.BytesIO()
            img.save(img_io, format="JPEG")
            img_io.seek(0)
            doc.add_paragraph(f"Detection {i+1}")
            doc.add_picture(img_io, width=Inches(5))
            doc.add_paragraph("")

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    insp = (inspector_name or "anonymous").replace(" ", "_")
    cls = (class_number or "unknownclass").replace(" ", "_")
    fname = f"{today_str}_{insp}_{cls}_report.docx"

    os.makedirs("temp_reports", exist_ok=True)
    local_path = os.path.join("temp_reports", fname)
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    doc.save(local_path)
    return bio, fname, local_path

# -----------------------------------------------------------
# 4️⃣  Header & Logo
# -----------------------------------------------------------
logo_b64 = get_image_base64("ASU-logo.png")
st.markdown(f"<img src='data:image/png;base64,{logo_b64}' width='250'/>", unsafe_allow_html=True)
st.title("AI‑Powered Classroom Inspection · ASU Edition")
st.markdown("Upload classroom images to detect key features and generate a report.")

# -----------------------------------------------------------
# 5️⃣  Image Uploader
# -----------------------------------------------------------
st.subheader("Step 1: Upload Classroom Images")
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
    if st.button("🗑️ Clear Uploaded Images"):
        st.session_state.uploaded_files = []
        new_id = int(st.session_state.uploader_key.split("_")[1]) + 1
        st.session_state.uploader_key = f"uploader_{new_id}"
        st.rerun()

# -----------------------------------------------------------
# 6️⃣  Inspection Details
# -----------------------------------------------------------
st.subheader("Step 1.5: Inspection Details")
col_a, col_b = st.columns(2)
with col_a:
    class_number = st.text_input("Classroom Number (e.g., 'DH 101')", value=st.session_state.get("class_number", ""), key="class_number")
with col_b:
    inspector_options = ["Nitin", "Jose", "Priyam", "Tanvi", "Others"]
    inspector_name = st.selectbox("Inspector Name", inspector_options, index=inspector_options.index(st.session_state.inspector_name) if st.session_state.inspector_name in inspector_options else 0, key="inspector_name")
custom_name = ""
if inspector_name == "Others":
    custom_name = st.text_input("Enter inspector name", value=st.session_state.get("custom_inspector_name", ""), key="custom_inspector_name")
inspector_used = custom_name.strip() if inspector_name == "Others" else inspector_name
inspector_used = inspector_used or "anonymous"

# -----------------------------------------------------------
# 7️⃣  Model & YOLO Options
# -----------------------------------------------------------
st.subheader("Step 2: Choose Model & Options")
model_options = ["Best (faster, lower cost)", "Basic (fastest, cheapest)", "Expert (most advanced reasoning) – need to add"]
model_choice = st.selectbox("LLM Model:", model_options, index=model_options.index(st.session_state.model_choice) if st.session_state.model_choice in model_options else model_options.index(DEFAULT_MODEL), key="model_choice")

enable_yolo = st.checkbox("Detect and highlight key objects?", value=st.session_state.enable_yolo, key="enable_yolo")
selected_model, model_comment = {"Best (faster, lower cost)": ("gpt-4o", "Using best model."), "Basic (fastest, cheapest)": ("gpt-4o-mini", "Using smaller model."), "Expert (most advanced reasoning) – need to add": ("gpt-4o", "Using expert model.")}[model_choice]

if enable_yolo:
    yolo_model = load_yolo_cached()
    YOLO_CLASSES = {
        0: "911 Address",
        1: "Bill of Rights Constitution",
        2: "Capacity Sign",
        3: "Classroom Layout",
        4: "Classroom Support Pocket",
        5: "Clock",
        6: "Dirty Whiteboard",
        7: "ERG Layout",
        8: "Exit Sign",
        9: "Flag",
        10: "Miscellaneous Objects",
        11: "No Food/Drinks Sign",
        12: "Recycle Bin",
        13: "Scrapes",
        14: "Stains",
        15: "Trash Bin",
        16: "University Classrooms Pocket",
        17: "Whiteboard",
        18: "Window Covering",
    }
    def detect_objects(images):
        counts, annotated = {}, []
        for img_file in images:
            img = Image.open(img_file).convert("RGB")
            results = yolo_model(np.array(img), classes=list(YOLO_CLASSES.keys()))
            for result in results:
                if not result.boxes:
                    continue
                found=False
                for box in result.boxes:
                    cls=int(box.cls[0])
                    if cls in YOLO_CLASSES:
                        label=YOLO_CLASSES[cls]
                        counts[label]=counts.get(label,0)+1
                        found=True
                if found:
                    annotated.append(Image.fromarray(result.plot(conf=True, labels=True)))
        return counts, annotated
else:
    detect_objects=None

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
11. Clock: Present/Absent.
12. Additional Comments: Any unusual items or safety issues.
"""
prompt_default = build_default_prompt(enable_yolo)
with st.expander("⚙️ More Options: Edit Inspection Prompt"):
    prompt = st.text_area("LLM Prompt", prompt_default, height=260)

# -----------------------------------------------------------
# 🔟  Run Inspection
# -----------------------------------------------------------
center = st.columns([1,2,1])[1]
run_btn = center.button("🔍 Run Inspection", use_container_width=True)
status = st.empty()

if run_btn:
    if not st.session_state.uploaded_files:
        st.error("Please upload at least one image.")
        st.stop()

    status.info("Preparing images…")
    with st.expander("📷 View Uploaded Images"):
        cols=st.columns(min(len(st.session_state.uploaded_files),4))
        for i,f in enumerate(st.session_state.uploaded_files):
            with cols[i%4]:
                st.image(Image.open(f), caption=f"Image {i+1}", use_container_width=True)

    detections, annotated_imgs = {}, []
    if enable_yolo and detect_objects:
        status.info("Detecting objects with YOLO…")
        detections, annotated_imgs = detect_objects(st.session_state.uploaded_files)
        st.write("**Detected objects:**", detections or "No objects detected.")
        if annotated_imgs:
            with st.expander("📦 YOLO Detections"):
                cols=st.columns(min(len(annotated_imgs),4))
                for i,im in enumerate(annotated_imgs):
                    with cols[i%4]:
                        st.image(im, caption=f"Detection {i+1}", use_container_width=True)

    status.info("Analyzing classroom with AI Vision…")
    report = call_gpt_hybrid(
        st.session_state.uploaded_files,
        prompt,
        selected_model,
        detection_data=detections if enable_yolo else None,
    )

    st.subheader("📝 Inspection Report")
    st.markdown(report)

    bio, fname, local_path = generate_docx_report(
        report,
        st.session_state.uploaded_files,
        annotated_images=annotated_imgs,
        class_number=class_number,
        inspector_name=inspector_used,
    )
    st.download_button("📄 Download DOCX Report", data=bio, file_name=fname)

    status.success("All done! 🎉")

# -----------------------------------------------------------
# 1️⃣1️⃣ Sidebar – About
# -----------------------------------------------------------
with st.sidebar.expander("📄 About This Project"):
    avatar_b64 = get_image_base64("musk-photo-1.jpg")
    st.markdown(
        f"""
        <div style='text-align:center;'>
            <img src='data:image/jpeg;base64,{avatar_b64}' style='width:150px;border-radius:50%;'/>
            <div style='color:#8C1D40;font-size:16px;margin-top:8px;'><strong>Nitin Reddy Yarava</strong></div>
            <p style='font-size:16px;'>
                This project automates ASU classroom inspections using a custom YOLO model and GPT‑4 Vision.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------
# 1️⃣2️⃣ Footer
# -----------------------------------------------------------
st.markdown("---")
st.caption("Built by Nitin, a CS student at ASU ✌️")
