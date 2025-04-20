import base64
import datetime
import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from docx import Document
from docx.shared import Inches
from ultralytics import YOLO
import openai

# 0️⃣ ENV & API KEYS -----------------------------------------------------------
load_dotenv(r"C:\Users\yniti\Downloads\classroom_inspector_api_key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1️⃣ PAGE CONFIG & STYLING ---------------------------------------------------
st.set_page_config(page_title="AI Classroom Inspector", layout="wide")

CSS = """
<style>
  .block-container {padding:2rem;background:#fff;color:#000}
  .stButton>button{color:#fff;background:#8C1D40;border:none;padding:.5rem 1rem;font-size:16px;border-radius:8px}
  .stButton>button:hover{background:#FFC627;color:#000}
  label, .stSubheader, .stCaption{color:#8C1D40!important}
  input, textarea{background:#fff8dc;border:1px solid #8C1D40;color:#000}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# 2️⃣ SESSION DEFAULTS --------------------------------------------------------
DEFAULTS = {"inspector_name":"Nitin","model_choice":"Basic (fastest, cheapest)","enable_yolo":True}
for k,v in DEFAULTS.items():
    st.session_state.setdefault(k,v)

# 3️⃣ HELPERS -----------------------------------------------------------------

def b64_file(path: str|Path):
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode()

def b64_image(file):
    buf=io.BytesIO(); Image.open(file).convert("RGB").save(buf,format="JPEG"); return base64.b64encode(buf.getvalue()).decode()

@st.cache_resource(show_spinner="Loading YOLO model …")
def load_yolo(): return YOLO("yolov8n.pt")

# GPT‑Vision

def vision_chat(imgs,prompt,model,anom=None):
    blocks=[{"type":"text","text":prompt}]
    if st.session_state.enable_yolo:
        blocks.insert(1,{"type":"text","text":"No anomalies detected." if not anom else "Detected anomalies:\n"+"\n".join(f"- {k}: {v}" for k,v in anom.items())})
    blocks+=[{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64_image(i)}"}} for i in imgs]
    rsp=openai.OpenAI().chat.completions.create(model=model,messages=[{"role":"user","content":blocks}],max_tokens=1000,temperature=0.2,top_p=0.8)
    return rsp.choices[0].message.content

# DOCX builder

def build_docx(text,orig,ann,cls,insp):
    d=Document(); d.add_heading("Classroom Inspection Report",0)
    d.add_paragraph(f"Class Number: {cls or '__________________'}")
    d.add_paragraph(f"Inspector: {insp or '__________________'}")
    d.add_paragraph(f"Date: {datetime.date.today():%B %d, %Y}")
    d.add_paragraph(""); d.add_heading("1. Inspection Summary",level=1)
    for l in text.strip().split('\n'):
        if l.strip(): d.add_paragraph(l.strip(),style="List Bullet")
    d.add_heading("2. Uploaded Classroom Images",level=1)
    for i,f in enumerate(orig,1): buf=io.BytesIO(); Image.open(f).save(buf,format="JPEG"); buf.seek(0); d.add_paragraph(f"Original Image {i}"); d.add_picture(buf,width=Inches(5)); d.add_paragraph("")
    if ann:
        d.add_heading("3. YOLO Anomaly Detections",level=1)
        for i,img in enumerate(ann,1): buf=io.BytesIO(); img.save(buf,format="JPEG"); buf.seek(0); d.add_paragraph(f"Anomaly Image {i}"); d.add_picture(buf,width=Inches(5)); d.add_paragraph("")
    today=datetime.date.today().strftime("%Y-%m-%d"); fname=f"{today}_{(insp or 'anonymous').replace(' ','_')}_{(cls or 'unknownclass').replace(' ','_')}_report.docx"
    bio=io.BytesIO(); d.save(bio); bio.seek(0)
    os.makedirs("temp_reports",exist_ok=True); d.save(os.path.join("temp_reports",fname))
    return bio,fname

# 4️⃣ HEADER ---------------------------------------------------------------
logo=b64_file("ASU-logo.png"); st.markdown(f"<img src='data:image/png;base64,{logo}' width='220'/>",unsafe_allow_html=True)
st.title("AI‑Powered Classroom Inspection · ASU Edition")
st.markdown("Upload images, detect issues, generate a concise report.")

# 5️⃣ IMAGE UPLOAD ----------------------------------------------------------
if "uploader_key" not in st.session_state: st.session_state.uploader_key="uploader_0"
files=st.file_uploader("Upload images (JPG/PNG)",accept_multiple_files=True,type=["jpg","jpeg","png"],key=st.session_state.uploader_key)
if files: st.session_state.uploaded_files=files
if st.session_state.get("uploaded_files") and st.button("🗑️ Clear Images"):
    st.session_state.uploaded_files=[]; st.session_state.uploader_key=f"uploader_{int(st.session_state.uploader_key.split('_')[1])+1}"; st.rerun()

# 6️⃣ DETAILS ---------------------------------------------------------------
st.subheader("Step 1.5: Inspection Details")
cl1,cl2=st.columns(2)
with cl1:
    class_no=st.text_input("Classroom Number (e.g., DH 101)",value=st.session_state.get("class_no","") ,key="class_no")
with cl2:
    insp_opts=["Nitin","Jose","Priyam","Tanvi","Others"]
    insp_sel=st.selectbox("Inspector Name",insp_opts,index=insp_opts.index(st.session_state.inspector_name) if st.session_state.inspector_name in insp_opts else 0,key="inspector_name")
custom_insp=""; 
if insp_sel=="Others": custom_insp=st.text_input("Enter inspector name",value=st.session_state.get("custom_insp","") ,key="custom_insp")
inspector_final=(custom_insp.strip() if insp_sel=="Others" else insp_sel) or "anonymous"

# 7️⃣ MODEL & YOLO -----------------------------------------------------------
st.subheader("Step 2: Model & Options")
mods=["Best (faster, lower cost)","Basic (fastest, cheapest)","Expert (most advanced reasoning for images) – need to add"]
model_choice=st.selectbox("LLM Model",mods,index=mods.index(st.session_state.model_choice) if st.session_state.model_choice in mods else 1,key="model_choice")
use_yolo=st.checkbox("Detect anomalies with YOLO?",value=st.session_state.enable_yolo,key="enable_yolo")
MODEL_MAP={mods[0]:("gpt-4o","Using best model."),mods[1]:("gpt-4o-mini","Using smaller model."),mods[2]:("gpt-4o","Using expert model.")}
sel_model,model_msg=MODEL_MAP[model_choice]

# 8️⃣ YOLO helper ------------------------------------------------------------
if use_yolo:
    yolo=load_yolo(); ANOM={0:"person",1:"bicycle",24:"backpack",25:"umbrella",26:"handbag",36:"skateboard",39:"bottle",40:"wine glass",41:"cup",42:"fork",43:"knife",44:"spoon",45:"bowl",46:"banana",47:"apple",48:"sandwich",49:"orange",50:"broccoli",51:"carrot",52:"hot dog",53:"pizza",54:"donut",55:"cake",67:"cell phone",73:"book",75:"vase",76:"scissors",78:"hair drier",79:"toothbrush"}
    def detect(imgs):
        counts,ann={},[]
        for f in imgs:
            res=yolo(Image.open(f).convert("RGB"),classes=list(ANOM.keys()))[0]
            if res.boxes and res.boxes.cls.numel()>0:
                for i in res.boxes.cls.cpu().numpy().astype(int): counts[ANOM.get(i,"unk")]=counts.get(ANOM.get(i,"unk"),0)+1
                ann.append(Image.fromarray(res.plot()[:,:,::-1]))
        return counts,ann
else:
    detect=lambda x:(None,None)

# 9️⃣ PROMPT ---------------------------------------------------------------

def default_prompt(y):
    return f"{model_msg}\nKeep each of these 13 items to ONE short sentence (or ‘No problems found.’){' Include anomaly counts.' if y else ''}."""

prompt=st.text_area("LLM Prompt",default_prompt(use_yolo),height=180)

# 🔟 RUN --------------------------------------------------------------------
run=st.button("🔍 Run Inspection",type="primary")
if run:
    imgs=st.session_state.get("uploaded_files",[])
    if not imgs:
        st.error("Please upload images first."); st.stop()
    counts,annot=detect(imgs)
    st.info("Calling LLM…")
    report=vision_chat(imgs,prompt,sel_model,counts)
    st.success("Report generated ✅")

    st.subheader("Inspection Report"); st.markdown(report)
    bio,doc_name=build_docx(report,imgs,annot,class_no,inspector_final)
    st.download_button("📄 Download DOCX",data=bio,file_name=doc)


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
