import streamlit as st
import os
import re
import tempfile
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import subprocess
import shutil

# Optional conversion libs
try:
    from docx2pdf import convert as docx2pdf_convert  # Windows/macOS only
except ImportError:  # Linux fallback will raise later if used
    docx2pdf_convert = None

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
except ImportError:
    canvas = None  # type: ignore

# ───────────────────────────────────────────────────────────────
#  Streamlit page config
# ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Upload to Google Drive", layout="wide")
st.title("📂 Upload Reports to Google Drive")
st.markdown("Upload modified DOCX or TXT files and they'll be converted to **PDF** and stored in Google Drive based on the filename.")

st.divider()

# ───────────────────────────────────────────────────────────────
#  Session‑state helpers
# ───────────────────────────────────────────────────────────────
if "drive_uploaded_files" not in st.session_state:
    st.session_state.drive_uploaded_files = []
if "drive_uploader_key" not in st.session_state:
    st.session_state.drive_uploader_key = "drive_uploader_0"

# ───────────────────────────────────────────────────────────────
#  Google Drive service
# ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_drive_service():
    creds = None
    if os.path.exists("credentials.json"):
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", scopes=["https://www.googleapis.com/auth/drive.file"]
        )
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    elif "GOOGLE_CREDENTIALS" in st.secrets:
        creds = Credentials.from_authorized_user_info(st.secrets["GOOGLE_CREDENTIALS"])
    else:
        st.error("Google Drive credentials not found.")
        return None

    return build("drive", "v3", credentials=creds)

# ───────────────────────────────────────────────────────────────
#  Drive folder helpers
# ───────────────────────────────────────────────────────────────

def create_folder(service, name, parent_id=None):
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    folder = service.files().create(body=meta, fields="id").execute()
    return folder.get("id")


def get_or_create_folder_path(service, path: str):
    parts = path.strip("/").split("/")
    parent_id = None
    for part in parts:
        query = (
            f"name='{part}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        if parent_id:
            query += f" and '{parent_id}' in parents"
        res = service.files().list(q=query, fields="files(id, name)").execute()
        items = res.get("files", [])
        parent_id = items[0]["id"] if items else create_folder(service, part, parent_id)
    return parent_id

# ───────────────────────────────────────────────────────────────
#  Filename parsing → folder path
# ───────────────────────────────────────────────────────────────

def parse_filename(filename: str):
    # Expected: 2025-04-20_DH101_report.docx
    m = re.match(r"(\d{4}-\d{2}-\d{2})_([A-Za-z0-9_]+)_report\.(docx|txt)$", filename)
    if m:
        date, classroom, _ = m.groups()
        year, month, _ = date.split("-")
        return f"Classroom_Inspections/{year}/{month}/{classroom}"
    return None

# ───────────────────────────────────────────────────────────────
#  Conversion helpers
# ───────────────────────────────────────────────────────────────

def txt_to_pdf(txt_path: str, pdf_path: str):
    if canvas is None:
        raise RuntimeError("reportlab is required for TXT→PDF conversion but not installed.")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 72  # 1 inch margin top
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            c.drawString(72, y, line.rstrip("\n"))
            y -= 14
            if y < 72:
                c.showPage()
                y = height - 72
    c.save()


def convert_to_pdf(tmp_input: str, ext: str) -> str | None:
    """Return path to PDF if conversion succeeded, else None."""
    tmp_dir = tempfile.mkdtemp()
    tmp_pdf = os.path.join(
        tmp_dir, os.path.splitext(os.path.basename(tmp_input))[0] + ".pdf"
    )
    try:
        # Windows/macOS
        if ext == "docx" and docx2pdf_convert is not None:
            docx2pdf_convert(tmp_input, tmp_pdf)
            return tmp_pdf
        # Linux fallback via LibreOffice
        if ext == "docx" and docx2pdf_convert is None:
            if shutil.which("soffice"):
                subprocess.run(
                    ["soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp_dir, tmp_input],
                    check=True,
                )
                if os.path.exists(tmp_pdf):
                    return tmp_pdf
            else:
                st.warning("LibreOffice (`soffice`) not found; cannot convert DOCX → PDF.")
        # TXT → PDF
        if ext == "txt":
            txt_to_pdf(tmp_input, tmp_pdf)
            return tmp_pdf
    except Exception as e:
        st.warning(f"PDF conversion failed for {os.path.basename(tmp_input)} → {e}")
    return None

# ───────────────────────────────────────────────────────────────
#  Upload logic
# ───────────────────────────────────────────────────────────────

def upload_file(service, local_path: str, drive_folder_id: str):
    filename = os.path.basename(local_path)
    media = MediaFileUpload(local_path, mimetype="application/pdf")
    meta = {"name": filename, "parents": [drive_folder_id]}
    service.files().create(body=meta, media_body=media, fields="id").execute()

# ───────────────────────────────────────────────────────────────
#  Streamlit uploader widget
# ───────────────────────────────────────────────────────────────

uploaded_files = st.file_uploader(
    "📤 Upload DOCX or TXT reports (will convert to PDF automatically)",
    type=["docx", "txt"],
    accept_multiple_files=True,
    key=st.session_state.drive_uploader_key,
    help="Filename must follow YYYY-MM-DD_Classroom_report.docx format",
)

if uploaded_files:
    st.session_state.drive_uploaded_files = uploaded_files

if st.session_state.drive_uploaded_files:
    if st.button("🗑️ Clear Uploaded Files"):
        st.session_state.drive_uploaded_files = []
        new_id = int(st.session_state.drive_uploader_key.split("_")[-1]) + 1
        st.session_state.drive_uploader_key = f"drive_uploader_{new_id}"
        st.rerun()

# ───────────────────────────────────────────────────────────────
#  Main upload process
# ───────────────────────────────────────────────────────────────
if st.session_state.drive_uploaded_files:
    service = get_drive_service()
    if service is None:
        st.stop()

    for file in st.session_state.drive_uploaded_files:
        folder_path = parse_filename(file.name)
        if not folder_path:
            st.warning(f"⚠️ Filename `{file.name}` does not match expected format. Please rename and try again.")
            continue

        st.markdown(f"📁 Detected Folder: `{folder_path}`")
        dest_folder_id = get_or_create_folder_path(service, folder_path)

        # Save uploaded file to a temporary location first
        ext = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_in:
            tmp_in.write(file.read())
            tmp_in.flush()
            tmp_input_path = tmp_in.name

        pdf_path = convert_to_pdf(tmp_input_path, ext)
        if pdf_path is None:
            st.error(f"Failed to convert `{file.name}` to PDF. Uploading original file instead.")
            media = MediaFileUpload(tmp_input_path, mimetype="application/octet-stream")
            meta = {"name": file.name, "parents": [dest_folder_id]}
            service.files().create(body=meta, media_body=media, fields="id").execute()
            os.unlink(tmp_input_path)
            continue

        pdf_filename = os.path.basename(pdf_path)
        upload_file(service, pdf_path, dest_folder_id)
        st.markdown(f"✅ **{pdf_filename}** uploaded to `{folder_path}`")
        os.unlink(tmp_input_path)
        os.unlink(pdf_path)



st.markdown("---")
st.caption("Built by Nitin, a CS student at ASU ✌️")
