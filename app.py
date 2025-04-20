# app.py  — complete file  ✨

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
import openai
import torch  # still required for GPT vision; safe even if YOLO unavailable

# ── Attempt to load Ultralytics (may fail on libGL‑less Linux) ───────────────
YOLO_AVAILABLE = True
try:
    from ultralytics import YOLO
except Exception as e:          # ImportError OR missing shared‑library
    YOLO_AVAILABLE = False
    YOLO_IMPORT_ERR = str(e)

# ── API Keys ────────────────────────────────────────────────────────────────
load_dotenv(r"C:\Users\yniti\Downloads\classroom_inspector_api_key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── Streamlit page & styling ───────────────────────────────────────────────
st.set_page_config(page_title="AI Classroom Inspector", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {padding:2rem;background:#fff;color:#000;}
    .stButton>button{color:#fff;background:#8C1D40;border:none;padding:.5rem 1rem;font-size:16px;border-radius:8px;}
    .stButton>button:hover{background:#FFC627;color:#000;}
    .stSelectbox label,.stCheckbox label,.stTextArea label,
    .stSubheader,.stCaption,.stMarkdown,.stTextInput label{color:#8C1D40!important;}
    .stTextInput>div>input,.stTextArea textarea{background:#fff8dc;border:1px solid #8C1D40;color:#000;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session defaults (remember across reruns) ──────────────────────────────
DEFAULT_INSPECTOR = "Nitin"
DEFAULT_MODEL     = "Basic (fastest, cheapest)"
DEFAULT_YOLO      = True

if "inspector_name" not in st.session_state:
    st.session_state.inspector_name = DEFAULT_INSPECTOR
if "model_choice" not in st.session_state:
    st.session_state.model_choice = DEFAULT_MODEL
if "enable_yolo" not in st.session_state:
    st.session_state.enable_yolo = DEFAULT_YOLO

# ── Utility helpers ────────────────────────────────────────────────────────
def get_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def image_to_base64(image_file) -> str:
    img = Image.open(image_file).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

# Cache YOLO only if it’s actually available
if YOLO_AVAILABLE:
    @st.cache_resource(show_spinner="Loading YOLO model…")
    def load_yolo_cached():
        return YOLO("yolov8n.pt")     # tiny CPU‑friendly model

# GPT‑4o Vision wrapper
def call_gpt_hybrid(images, prompt, model, anomaly_data=None):
    blocks = [{"type": "text", "text": prompt}]
    if st.session_state.enable_yolo and anomaly_data is not None:
        if anomaly_data:
            lines = "\n".join(f"- {k}: {v}" for k, v in anomaly_data.items())
            blocks.insert(1, {"type":"text","text":f"Anomalies detected:\n{lines}"})
        else:
            blocks.insert(1, {"type":"text","text":"No anomalies detected by YOLO."})
    blocks += [
        {"type":"image_url",
         "image_url":{"url":f"data:image/jpeg;base64,{image_to_base64(im)}"}}
        for im in images
    ]
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":blocks}],
        max_tokens=1000,
        temperature=0.2,
        top_p=0.8,
    )
    return resp.choices[0].message.content

def generate_docx_report(report_text, original_images,
                         anomaly_images=None,
                         class_number=None,
                         inspector_name=None):
    doc = Document()
    doc.add_heading("Classroom Inspection Report", 0)
    doc.add_paragraph(f"Class Number: {class_number or '__________________'}")
    doc.add_paragraph(f"Inspector: {inspector_name or '__________________'}")
    doc.add_paragraph(f"Date: {datetime.date.today():%B %d, %Y}")
    doc.add_paragraph()

    doc.add_heading("1. Inspection Summary", level=1)
    for line in report_text.strip().splitlines():
        if line.strip():
            doc.add_paragraph(line.strip(), style="List Bullet")

    doc.add_heading("2. Uploaded Classroom Images", level=1)
    for i,img_file in enumerate(original_images,1):
        img = Image.open(img_file)
        bio = io.BytesIO(); img.save(bio,format="JPEG"); bio.seek(0)
        doc.add_paragraph(f"Original Image {i}")
        doc.add_picture(bio, width=Inches(5)); doc.add_paragraph()

    if anomaly_images:
        doc.add_heading("3. YOLO Anomaly Detections", level=1)
        for i,img in enumerate(anomaly_images,1):
            bio = io.BytesIO(); img.save(bio,format="JPEG"); bio.seek(0)
            doc.add_paragraph(f"Anomaly Image {i}")
            doc.add_picture(bio, width=Inches(5)); doc.add_paragraph()

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    inspector = (inspector_name or "anonymous").replace(" ","_")
    classroom  = (class_number or "unknownclass").replace(" ","_")
    filename   = f"{today_str}_{inspector}_{classroom}_report.docx"

    out = io.BytesIO(); doc.save(out); out.seek(0)
    os.makedirs("temp_reports", exist_ok=True)
    local_path = os.path.join("temp_reports", filename)
    with open(local_path,"wb") as fp: doc.save(fp)
    return out, filename, local_path

# ── Page header ───────────────────────────────────────────────────────────
logo_b64 = get_image_base64("ASU-logo.png")
st.markdown(f"<img src='data:image/png;base64,{logo_b64}' width='250'/>",
            unsafe_allow_html=True)
st.title("AI‑Powered Classroom Inspection · ASU Edition")
st.write("Upload classroom images to detect issues automatically and "
         "generate a concise inspection report.")

# ── Image uploader ────────────────────────────────────────────────────────
st.subheader("Step 1 · Upload Classroom Images")
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader_0"

uploaded_files = st.file_uploader(
    "Select one or more JPG / PNG images",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"],
    key=st.session_state.uploader_key,
)
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

if st.session_state.uploaded_files:
    if st.button("🗑️ Clear images"):
        st.session_state.uploaded_files = []
        new_id = int(st.session_state.uploader_key.split("_")[-1]) + 1
        st.session_state.uploader_key = f"uploader_{new_id}"
        st.rerun()

# ── Step 1.5 · Class & Inspector details ─────────────────────────────────
st.subheader("Step 1.5 · Inspection Details")
colA,colB = st.columns(2)
with colA:
    class_number = st.text_input(
        "Classroom number (e.g. DH 101) – optional:",
        value=st.session_state.get("class_number",""),
        key="class_number",
    )
with colB:
    options = ["Nitin","Jose","Priyam","Tanvi","Others"]
    inspector_sel = st.selectbox(
        "Inspector:",
        options,
        index=options.index(st.session_state.inspector_name)
              if st.session_state.inspector_name in options else 0,
        key="inspector_name",
    )
custom_name = ""
if inspector_sel=="Others":
    custom_name = st.text_input(
        "Enter inspector name:",
        value=st.session_state.get("custom_inspector_name",""),
        key="custom_inspector_name",
    )
inspector_used = (custom_name.strip() if inspector_sel=="Others" else inspector_sel) or "anonymous"

# ── Step 2 · Model + YOLO options ────────────────────────────────────────
st.subheader("Step 2 · Choose Model & Options")
model_options = [
    "Best (faster, lower cost)",
    "Basic (fastest, cheapest)",
    "Expert (most advanced reasoning for images) – need to add",
]
model_choice = st.selectbox(
    "LLM model:",
    model_options,
    index=model_options.index(st.session_state.model_choice)
          if st.session_state.model_choice in model_options
          else model_options.index(DEFAULT_MODEL),
    key="model_choice",
)

enable_yolo_checkbox = st.checkbox(
    "Detect and highlight unusual objects?",
    value=st.session_state.enable_yolo and YOLO_AVAILABLE,
    disabled=not YOLO_AVAILABLE,
    help="Disabled due to missing Ultralytics/OpenCV libraries." if not YOLO_AVAILABLE
         else "Toggle YOLO anomaly detection.",
    key="enable_yolo",
)
enable_yolo = enable_yolo_checkbox  # convenience

# ── Model mapping ────────────────────────────────────────────────────────
model_map = {
    "Best (faster, lower cost)": ("gpt-4o", "Using best model."),
    "Basic (fastest, cheapest)": ("gpt-4o-mini", "Using smaller model."),
    "Expert (most advanced reasoning for images) – need to add":
        ("gpt-4o","Using expert-level reasoning model (gpt‑4o)."),
}
selected_model, model_comment = model_map[model_choice]

# ── Build YOLO functions if enabled ──────────────────────────────────────
if YOLO_AVAILABLE and enable_yolo:
    yolo_model = load_yolo_cached()
    ANOMALY_CLASSES = {  # subset of COCO
        0:"person",1:"bicycle",24:"backpack",25:"umbrella",26:"handbag",
        36:"skateboard",39:"bottle",40:"wine glass",41:"cup",42:"fork",
        43:"knife",44:"spoon",45:"bowl",46:"banana",47:"apple",48:"sandwich",
        49:"orange",50:"broccoli",51:"carrot",52:"hot dog",53:"pizza",
        54:"donut",55:"cake",67:"cell phone",73:"book",75:"vase",
        76:"scissors",78:"hair drier",79:"toothbrush"
    }
    def detect_anomalies(images):
        counts, annotated = {}, []
        for f in images:
            img = Image.open(f).convert("RGB")
            res = yolo_model(np.array(img), classes=list(ANOMALY_CLASSES))
            for r in res:
                if not r.boxes: continue
                flag=False
                for box in r.boxes:
                    cls=int(box.cls[0])
                    if cls in ANOMALY_CLASSES:
                        name=ANOMALY_CLASSES[cls]
                        counts[name]=counts.get(name,0)+1
                        flag=True
                if flag:
                    annotated.append(Image.fromarray(r.plot(conf=True,labels=True)))
        return counts, annotated
else:
    detect_anomalies = None  # YOLO disabled

# ── Prompt template ──────────────────────────────────────────────────────
def build_prompt(use_yolo: bool)->str:
    extra = " and anomaly counts" if use_yolo else ""
    return f"""{model_comment}

You are a classroom inspection assistant. You will be given images{extra}.
⚠️ Keep each of the 13 items to **one very short sentence** (“No problems found.” if OK).

1. Side Walls (not ceiling)
2. Ceiling
3. Board
4. Floor
5. Number of Bins (gray=trash, blue=recycle)
6. Capacity Sign
7. Lights
8. Support & UCL Pocket
9. Flag
10. Food/Drinks Plaque
11. Instructor's Desk
12. Clock
13. Additional Comments
If something is unclear, respond “Cannot determine.”
"""
prompt_default = build_prompt(enable_yolo)
with st.expander("⚙️ Edit Prompt"):
    prompt = st.text_area("LLM prompt:", prompt_default, height=260)

# ── Run button ───────────────────────────────────────────────────────────
run_btn = st.button("🔍 Run Inspection", type="primary")
status = st.empty()

if run_btn:
    if not st.session_state.uploaded_files:
        st.error("Please upload at least one image."); st.stop()

    # Show input images
    status.info("Preparing images…")
    with st.expander("📷 Uploaded images:"):
        cols = st.columns(min(len(st.session_state.uploaded_files),4))
        for i,f in enumerate(st.session_state.uploaded_files):
            with cols[i%4]:
                st.image(Image.open(f), caption=f"Image {i+1}", use_container_width=True)

    # Optional YOLO
    annotated_imgs = anomalies = None
    if enable_yolo and detect_anomalies:
        status.info("Detecting anomalies with YOLO…")
        anomalies, annotated_imgs = detect_anomalies(st.session_state.uploaded_files)
        if annotated_imgs:
            with st.expander("📦 YOLO Detections"):
                cols=st.columns(min(len(annotated_imgs),4))
                for i,img in enumerate(annotated_imgs):
                    with cols[i%4]:
                        st.image(img, caption=f"Detections {i+1}", use_container_width=True)
        else:
            st.info("No anomalies detected.")

    # GPT call
    status.info("Calling AI model…")
    report_text = call_gpt_hybrid(
        st.session_state.uploaded_files,
        prompt,
        selected_model,
        anomalies,
    )
    status.success("Inspection report generated ✅")

    st.subheader("Inspection Report")
    st.markdown(report_text)

    # DOCX
    buf, docx_name, local_path = generate_docx_report(
        report_text,
        st.session_state.uploaded_files,
        annotated_imgs,
        class_number,
        inspector_used,
    )
    st.download_button(
        "📄 Download DOCX",
        data=buf,
        file_name=docx_name,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )
    b64 = base64.b64encode(report_text.encode()).decode()
    st.markdown(
        f"<a href='data:file/txt;base64,{b64}' download='{docx_name.replace('.docx','.txt')}'>📥 Download TXT</a>",
        unsafe_allow_html=True,
    )
    st.info(f"Report saved locally at {local_path}. Upload it via the "
            "'Upload to Drive' page when ready.")

# ── Sidebar /About ────────────────────────────────────────────────────────
with st.sidebar.expander("📄 About This Project"):
    avatar_b64 = get_image_base64("musk-photo-1.jpg")
    st.markdown(f"""
        <div style='text-align:center;'>
            <img src='data:image/jpeg;base64,{avatar_b64}' style='width:150px;border-radius:50%;'/>
            <div style='color:#8C1D40;font-size:16px;margin-top:8px;'><strong>Nitin Reddy Yarava</strong></div>
            <p style='font-size:16px;'>This project automates ASU classroom inspections using YOLOv8 for object detection and GPT‑4 Vision for reasoning.</p>
        </div>""", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built by Nitin, a CS student at ASU ✌️")
