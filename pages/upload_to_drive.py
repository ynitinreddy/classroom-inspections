import streamlit as st
import os, re, tempfile
from io import BytesIO
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# ── optional local‑conversion libs ─────────────────────────────
try:
    from docx2pdf import convert as docx2pdf_convert    # only works on Windows/macOS
except ImportError:
    docx2pdf_convert = None

try:
    from reportlab.pdfgen import canvas                 # TXT → PDF
    from reportlab.lib.pagesizes import letter
except ImportError:
    canvas = None  # TXT conversion will fail gracefully

# ── Streamlit page config ─────────────────────────────────────
st.set_page_config(page_title="Upload to Google Drive", layout="wide")
st.title("📂 Upload Reports to Google Drive")
st.markdown(
    "Upload **DOCX** or **TXT** files – they’re converted to **PDF** "
    "and stored in Google Drive under the proper classroom folder."
)
st.divider()

# ── session‑state init ────────────────────────────────────────
if "drive_uploaded_files" not in st.session_state:
    st.session_state.drive_uploaded_files = []
if "drive_uploader_key" not in st.session_state:
    st.session_state.drive_uploader_key = "drive_uploader_0"

# ── Google Drive service helper ───────────────────────────────
@st.cache_resource
def get_drive_service():
    creds = None
    if os.path.exists("credentials.json"):
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json",
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as tk:
            tk.write(creds.to_json())
    elif "GOOGLE_CREDENTIALS" in st.secrets:
        creds = Credentials.from_authorized_user_info(st.secrets["GOOGLE_CREDENTIALS"])
    else:
        st.error("Google Drive credentials not found.")
        return None

    return build("drive", "v3", credentials=creds)

# ── folder helpers ────────────────────────────────────────────
def create_folder(svc, name, parent=None):
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent:
        meta["parents"] = [parent]
    return svc.files().create(body=meta, fields="id").execute()["id"]

def get_or_create_folder_path(svc, path):
    parent = None
    for part in path.strip("/").split("/"):
        q = (
            f"name='{part}' and mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false"
        )
        if parent:
            q += f" and '{parent}' in parents"
        res = svc.files().list(q=q, fields="files(id)").execute()
        items = res.get("files", [])
        parent = items[0]["id"] if items else create_folder(svc, part, parent)
    return parent

# ── filename → Drive path ─────────────────────────────────────
def parse_filename(fname: str):
    # Accepts: YYYY-MM-DD_Inspector_Classroom_report.docx  (or .txt)
    m = re.match(
        r"(\\d{4}-\\d{2}-\\d{2})_([A-Za-z0-9_]+)_([A-Za-z0-9_]+)_report\\.(docx|txt)$",
        fname,
    )
    if not m:
        return None
    date, inspector, classroom, _ = m.groups()
    year, month, _ = date.split("-")
    subfolder = f"{inspector}_{classroom}"  # e.g. Nitin_DH_101
    return f"Classroom_Inspections/{year}/{month}/{subfolder}"


# ── TXT → PDF (local) ─────────────────────────────────────────
def txt_to_pdf(txt_path, pdf_path):
    if canvas is None:
        raise RuntimeError("reportlab not available")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 72
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            c.drawString(72, y, line.rstrip())
            y -= 14
            if y < 72:
                c.showPage()
                y = height - 72
    c.save()

# ── Google Drive helper uploads ───────────────────────────────
def upload_original(svc, local_path, folder_id):
    media = MediaFileUpload(local_path, resumable=True)
    meta = {"name": os.path.basename(local_path), "parents": [folder_id]}
    svc.files().create(body=meta, media_body=media).execute()

def export_gdoc_to_pdf(svc, file_id) -> bytes:
    return svc.files().export_media(fileId=file_id, mimeType="application/pdf").execute()

def save_pdf_blob(svc, pdf_blob, pdf_name, folder_id):
    fh = BytesIO(pdf_blob)
    media = MediaIoBaseUpload(fh, mimetype="application/pdf")
    meta = {"name": pdf_name, "parents": [folder_id]}
    svc.files().create(body=meta, media_body=media).execute()

# ── Streamlit uploader ────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📤 Upload DOCX or TXT reports",
    type=["docx", "txt"],
    accept_multiple_files=True,
    key=st.session_state.drive_uploader_key,
)

if uploaded_files:
    st.session_state.drive_uploaded_files = uploaded_files

if st.session_state.drive_uploaded_files:
    if st.button("🗑️ Clear list"):
        st.session_state.drive_uploaded_files = []
        # drive_uploader_key is like "drive_uploader_3".
        # Take the FINAL chunk, not the second one (robust if name ever changes).
        kid = int(st.session_state.drive_uploader_key.split("_")[-1]) + 1
        st.session_state.drive_uploader_key = f"drive_uploader_{kid}"

        st.rerun()

# ── main loop ─────────────────────────────────────────────────
if st.session_state.drive_uploaded_files:
    svc = get_drive_service()
    if svc is None:
        st.stop()

    for uf in st.session_state.drive_uploaded_files:
        path = parse_filename(uf.name)
        if not path:
            st.warning(f"⚠️ Filename `{uf.name}` doesn’t match convention.")
            continue

        st.markdown(f"**Folder** → `{path}`")
        folder_id = get_or_create_folder_path(svc, path)

        ext = uf.name.rsplit(".", 1)[1].lower()
        pdf_name = uf.name.rsplit(".", 1)[0] + ".pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(uf.read())
            tmp.flush()
            local_in = tmp.name

        pdf_blob = None

        # ── local attempt ──────────────────────────────────────
        try:
            if ext == "docx" and docx2pdf_convert and os.name != "posix":
                tmp_pdf = tempfile.mktemp(suffix=".pdf")
                docx2pdf_convert(local_in, tmp_pdf)
                pdf_blob = open(tmp_pdf, "rb").read()
                os.unlink(tmp_pdf)
            elif ext == "txt":
                tmp_pdf = tempfile.mktemp(suffix=".pdf")
                txt_to_pdf(local_in, tmp_pdf)
                pdf_blob = open(tmp_pdf, "rb").read()
                os.unlink(tmp_pdf)
        except Exception as e:
            st.info(f"Local conversion failed: {e}")

        # ── Google‑Drive fallback for DOCX ─────────────────────
        if pdf_blob is None and ext == "docx":
            meta = {
                "name": uf.name,
                "mimeType": "application/vnd.google-apps.document",
                "parents": [folder_id],
            }
            media = MediaFileUpload(
                local_in,
                mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            gdoc_id = svc.files().create(body=meta, media_body=media, fields="id").execute()["id"]
            try:
                pdf_blob = export_gdoc_to_pdf(svc, gdoc_id)
            except Exception as e:
                st.error(f"Drive conversion failed: {e}")

        # ── upload result ─────────────────────────────────────
        if pdf_blob:
            save_pdf_blob(svc, pdf_blob, pdf_name, folder_id)
            st.success(f"✅ Uploaded `{pdf_name}`")
        else:
            upload_original(svc, local_in, folder_id)
            st.warning(f"Uploaded original `{uf.name}` (no PDF generated)")

        os.unlink(local_in)

st.markdown("---")
st.caption("Built by Nitin, a CS student at ASU ✌️")
