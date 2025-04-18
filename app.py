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
torch.classes.__path__ = []

if "OPENAI_API_KEY" in st.secrets:          # Streamlit Cloud path
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:                                       # local dev path
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=r"C:\Users\yniti\Downloads\classroom_inspector_api_key.env")
    openai.api_key = os.getenv("OPENAI_API_KEY")



# Load the .env from a custom path
# load_dotenv(dotenv_path=r"C:\Users\yniti\Downloads\classroom_inspector_api_key.env")
# openai.api_key = os.getenv("OPENAI_API_KEY")

# openai.api_key = st.secrets["OPENAI_API_KEY"]

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
        background-color: #ffffff; /* White background */
        color: #000000; /* Default text black */
    }
    .stButton>button {
        color: white;
        background-color: #8C1D40; /* ASU Maroon */
        border: none;
        padding: 0.5rem 1rem;
        font-size: 16px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #FFC627; /* ASU Gold */
        color: black;
    }
    .stSelectbox label, .stCheckbox label, .stTextArea label, .stSubheader, .stCaption, .stMarkdown, .stTextInput label {
        color: #8C1D40 !important; /* ASU Maroon */
    }
    .stTextInput>div>input {
        background-color: #fff8dc; /* very light gold */
        border: 1px solid #8C1D40; /* Maroon border */
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


def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
# -----------------------------------------------------------
# 1.  Encapsulate model loading in a cached function
# -----------------------------------------------------------

# Put this near the other top‑level defs (anywhere before you need the model)
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_yolo():
    """
    Downloads yolov8n.pt the first time and keeps the model
    resident in memory across Streamlit reruns.
    """
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")      # tiny 6 MB model, good for CPU


img_b64 = get_image_base64("musk-photo-1.jpg")

st.image("ASU-logo.png", width=250)
st.title("AI-Powered Classroom Inspection - ASU Edition")
st.markdown("Welcome! Upload classroom images to automatically detect issues and generate an inspection report.")


# -----------------------------------------------------------
# 2.  Generate docx report
# -----------------------------------------------------------


def generate_docx_report(report_text, original_images, anomaly_images=None, class_number=None):

    doc = Document()

    # --- Title and Date ---
    doc.add_heading('Classroom Inspection Report', 0)
    if class_number:
        doc.add_paragraph(f"Class Number: {class_number}")
    else:
        doc.add_paragraph("Class Number: __________________")  # Leave blank if not provided
  # Leave for user to fill
    doc.add_paragraph(f"Date: {datetime.date.today().strftime('%B %d, %Y')}")
    doc.add_paragraph("")  # spacer

    # --- Report Text ---
    doc.add_heading('1. Inspection Summary', level=1)
    for line in report_text.strip().split('\n'):
        if line.strip():
            doc.add_paragraph(line.strip(), style='List Bullet')

    # --- Original Images ---
    doc.add_heading('2. Uploaded Classroom Images', level=1)
    for i, img_file in enumerate(original_images):
        img = Image.open(img_file)
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        doc.add_paragraph(f"Original Image {i+1}")
        doc.add_picture(img_io, width=Inches(5))
        doc.add_paragraph("")  # spacer

    # --- YOLO Anomaly Images (Optional) ---
    if anomaly_images:
        doc.add_heading('3. YOLO Anomaly Detections', level=1)
        for i, img in enumerate(anomaly_images):
            img_io = io.BytesIO()
            img.save(img_io, format='JPEG')
            img_io.seek(0)
            doc.add_paragraph(f"Anomaly Image {i+1}")
            doc.add_picture(img_io, width=Inches(5))
            doc.add_paragraph("")  # spacer

    # --- Save to BytesIO for download ---
    output_io = io.BytesIO()
    doc.save(output_io)
    output_io.seek(0)
    return output_io




# --- About Section ---
with st.sidebar.expander("📄 About This Project"):
    # st.image("musk-photo-1.jpg", width=150, caption="Nitin Reddy Yarava")
    # st.markdown("""
    #     <!-- <h3 style="color:#8C1D40;">About the Creator</h3> -->
    #     <p style="font-size: 16px;">
    #     This project was developed as part of an initiative to automate and improve classroom inspections at ASU. It uses a hybrid approach of YOLOv8 for object detection and GPT-4 Vision for smart visual reasoning. Built by a CS student passionate about ML-driven operations and automation.
    #     </p>
    #     """, unsafe_allow_html=True)
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{img_b64}" style="width:150px; border-radius: 50%;" />
            <div style="color:#8C1D40; font-size: 16px; margin-top: 8px;"><strong>Nitin Reddy Yarava</strong></div>
            <p style="font-size: 16px;">
        This project was developed as part of an initiative to automate and improve classroom inspections at ASU. It uses a hybrid approach of YOLOv8 for object detection and GPT-4 Vision for smart visual reasoning. Built by a CS student passionate about ML-driven operations and automation.
        </p>
        </div>
        """, unsafe_allow_html=True)

# --- Image Upload ---
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

# Append newly uploaded files
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files


if st.session_state.uploaded_files:
    if st.button("🗑️ Clear Uploaded Images"):
        st.session_state.uploaded_files = []
        # Change the uploader key to reset the widget
        key_id = int(st.session_state.uploader_key.split("_")[1]) + 1
        st.session_state.uploader_key = f"uploader_{key_id}"
        st.rerun()


# --- Optional Class Number Input ---
st.subheader("Step 1.5: Enter Class Number (Optional)")
class_number = st.text_input(
    "If you want, enter the classroom number (e.g., 'DH 101'). This will appear in the report and file name.",
    value=""
)



# --- Model Selection ---
st.subheader("Step 2: Choose Model & Options")
model_choice = st.selectbox(
    "Select LLM model type:",
    ["Best (faster, lower cost)", "Basic (fastest, cheapest)", "Expert (most advanced reasoning for images) - need to add"],
    index=0,
    help="Best uses GPT-4o; Basic uses GPT-40-mini; Expert uses the best available vision reasoning, need to add yet"
)
enable_yolo = st.checkbox(
    "Detect and highlight unusual objects?",
    value=False,
    help="Toggle to run or skip the YOLO-based anomaly object detector"
)

model_map = {
    "Best (faster, lower cost)": ("gpt-4o", "Using best model."),
    "Basic (fastest, cheapest)": ("gpt-4o-mini", "Using smaller model."),
    "Expert (most advanced reasoning for images) - need to add": ("gpt-4o", "Using expert-level reasoning model (gpt-4o).")
}
selected_model, model_comment = model_map[model_choice]

# Only load YOLO if requested
if enable_yolo:
    yolo_model = load_yolo()
    ANOMALY_CLASSES = {
        0: "person", 1: "bicycle", 24: "backpack", 25: "umbrella", 26: "handbag",
        36: "skateboard",
        39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
        44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
        49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
        54: "donut", 55: "cake", 67: "cell phone",
        73: "book", 75: "vase", 76: "scissors",
        78: "hair drier", 79: "toothbrush"
    }

def image_to_base64(image_file):
    img = Image.open(image_file).convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def detect_anomalies(images):
    counts = {}
    annotated_images = []

    for image_file in images:
        img = Image.open(image_file).convert("RGB")
        img_np = np.array(img)
        results = yolo_model(img_np)

        for result in results:
            result_img = result.plot(conf=True, labels=True)
            annotated_images.append(Image.fromarray(result_img))

            for box in result.boxes:
                cls = int(box.cls[0])
                if cls in ANOMALY_CLASSES:
                    name = ANOMALY_CLASSES[cls]
                    counts[name] = counts.get(name, 0) + 1

    return counts, annotated_images



# --- Prompt Section (Expandable) ---
default_prompt = f"""
{model_comment}

You are a classroom inspection assistant. You will be given images{ " and anomaly counts" if enable_yolo else "" }.
⚠️ VERY IMPORTANT: Keep each of the 13 items to one very short sentence (“No problems found.” if OK).

Use the numbered list 1 through 13. For each item, begin with the heading (e.g., “Walls:”), then give your observation very short. If nothing is wrong or noteworthy, simply respond with “No problems found.”

Only report issues that are clearly visible. If something is unclear, say “Cannot determine.”
1. Side Walls (not ceiling): Scuffs, scrapes, holes, Unsure?
2. Ceiling: Holes, stains, Unsure etc?
3. Board: Clean, Writings, or dirty, Unsure ?
4. Floor: Trash, stains, frayed tiles, tears, Unsure ?
5. No. of Bins: Count and type (gray is trash, blue is recycle), Unsure ?
6. Capacity Sign: Present or absent? If present, show the number, Unsure .
7. Lights: Are all working? If not, how many are out, Unsure ?
8. Support & UCL Pocket: Present or not, Unsure ?
9. Flag: Present or not, Unsure ?
10. Food/Drinks Plaque: Present or not, Unsure ?
11. Instructor's Desk: Visible or not. if visible, clean or not ?
12. Clock: Present or not, Unsure ?
13. Additional Comments: What are the Unusual Stuff found or seen in class if any?, etc.
"""

with st.expander("⚙️ More Options: Edit Inspection Prompt"):
    prompt = st.text_area("LLM Prompt", default_prompt, height=250)

# --- GPT Vision Call ---
def call_gpt_hybrid(images, prompt, model, anomaly_data=None):
    base64_images = [image_to_base64(img) for img in images]
    content_blocks = [{"type": "text", "text": prompt}]
    if enable_yolo:
        if anomaly_data:
            summary = "\n".join(f"- {k}: {v}" for k, v in anomaly_data.items())
            content_blocks.insert(1, {"type": "text", "text": f"These anomalies were detected by an object detection model:\n{summary}"})
        else:
            content_blocks.insert(1, {"type": "text", "text": "No anomalies were detected by the model."})
    content_blocks += [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in base64_images
    ]
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content_blocks}],
        max_tokens=1000,
        temperature=0.2,      # ↓ from the default 1.0
        top_p=0.8             # optional but usually helps
    )

    return resp.choices[0].message.content

# --- Submit Button ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_button = st.button("🔍 Run Inspection", use_container_width=True)

status = st.empty()

if run_button:
    anomaly_images = None
    if not st.session_state.uploaded_files:
        st.error("Please upload at least 1 image.")
    else:
        status.info("Preparing images…")
        with st.expander("📷 View uploaded images"):
            cols = st.columns(min(len(st.session_state.uploaded_files), 4))
            for i, f in enumerate(st.session_state.uploaded_files):
                with cols[i % 4]:
                    img = Image.open(f)
                    st.image(img, caption=f"Image {i+1}", use_container_width=True)

        anomalies = None
        if enable_yolo:
            status.info("Detecting anomalies with YOLO…")
            anomalies, annotated_images = detect_anomalies(st.session_state.uploaded_files)
            anomaly_images = annotated_images 


            with st.expander("📦 YOLO Anomaly Detections"):
                cols = st.columns(min(len(annotated_images), 4))
                for i, img in enumerate(annotated_images):
                    with cols[i % 4]:
                        st.image(img, caption=f"Detections in Image {i+1}", use_container_width=True)


        status.info("Calling AI Model…")
        report = call_gpt_hybrid(st.session_state.uploaded_files, prompt, selected_model, anomalies)
        status.success("Inspection report generated ✅")

        st.subheader("Inspection Report")
        st.markdown(report)
        # --- Generate DOCX ---
        docx_file = generate_docx_report(
            report,
            st.session_state.uploaded_files,
            anomaly_images if enable_yolo else None,
            class_number
        )

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        file_suffix = f"{today_str}_{class_number.replace(' ', '_')}" if class_number else "classroom_inspection_report"
        file_name = f"{file_suffix}.docx"

        # --- Download Button ---
        st.download_button(
            label="📄 Download Full Report (.docx)",
            data=docx_file,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )


        b64 = base64.b64encode(report.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="inspection_report.txt">📥 Download Report as TXT</a>'
        st.markdown(href, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("Built by Nitin, a CS student at ASU ✌️")