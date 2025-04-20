import streamlit as st
import os
import re
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# --- Page Config ---
st.set_page_config(page_title="Upload to Google Drive", layout="wide")
st.title("📂 Upload Reports to Google Drive")
st.markdown("Upload modified DOCX or TXT files, and they will be stored in Google Drive based on the filename.")

st.divider()

# --- Session State for Uploaded Files ---
if "drive_uploaded_files" not in st.session_state:
    st.session_state.drive_uploaded_files = []

if "drive_uploader_key" not in st.session_state:
    st.session_state.drive_uploader_key = "drive_uploader_0"

# --- Google Drive Auth ---
@st.cache_resource
def get_drive_service():
    creds = None
    if os.path.exists("credentials.json"):
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json",
            scopes=["https://www.googleapis.com/auth/drive.file"]
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

# --- Folder Creation ---
def create_folder(service, name, parent_id=None):
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder"
    }
    if parent_id:
        metadata["parents"] = [parent_id]
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder.get("id")

def get_or_create_folder_path(service, path):
    folders = path.strip("/").split("/")
    parent_id = None
    for folder in folders:
        query = f"name='{folder}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get("files", [])
        if files:
            parent_id = files[0]["id"]
        else:
            parent_id = create_folder(service, folder, parent_id)
    return parent_id

# --- Upload File ---
def upload_file(service, file, folder_id):
    metadata = {"name": file.name, "parents": [folder_id]}
    media = MediaFileUpload(file, mimetype="application/octet-stream")
    service.files().create(body=metadata, media_body=media, fields="id").execute()

# --- Parse Filename Format ---
def parse_filename(filename):
    # Format: 2025-04-20_DH101_report.docx
    pattern = r"(\d{4}-\d{2}-\d{2})_([A-Za-z0-9]+)_report\.(docx|txt)"
    match = re.match(pattern, filename)
    if match:
        date, classroom, _ = match.groups()
        year, month, _ = date.split("-")
        return f"Classroom_Inspections/{year}/{month}/{classroom}"
    return None

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "📤 Upload modified DOCX or TXT files",
    type=["docx", "txt"],
    accept_multiple_files=True,
    key=st.session_state.drive_uploader_key,
    help="Files must follow this format: YYYY-MM-DD_Classroom_report.docx"
)

if uploaded_files:
    st.session_state.drive_uploaded_files = uploaded_files

# --- Clear Button ---
if st.session_state.drive_uploaded_files:
    if st.button("🗑️ Clear Uploaded Files"):
        st.session_state.drive_uploaded_files = []
        key_id = int(st.session_state.drive_uploader_key.split("_")[1]) + 1
        st.session_state.drive_uploader_key = f"drive_uploader_{key_id}"
        st.rerun()

# --- Upload Process ---
if st.session_state.drive_uploaded_files:
    service = get_drive_service()
    if service:
        for file in st.session_state.drive_uploaded_files:
            folder_path = parse_filename(file.name)
            if folder_path:
                st.markdown(f"📁 Detected Folder: {folder_path}")
                folder_id = get_or_create_folder_path(service, folder_path)
                upload_file(service, file, folder_id)
                st.success(f"✅ Uploaded {file.name} to Google Drive in {folder_path}")
            else:
                st.warning(f"⚠️ Filename {file.name} does not match expected format. Please rename and try again.")

st.markdown("---")
st.caption("Built by Nitin, a CS student at ASU ✌️")