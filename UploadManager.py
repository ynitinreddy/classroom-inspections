from drive_utils import get_drive_service, get_or_create_folder, upload_file_to_drive

# ...
if submitted:
    service = get_drive_service()

    # --- Build Folder Hierarchy ---
    root_folder = get_or_create_folder(service, "ClassroomReports")
    class_folder = get_or_create_folder(service, class_input.replace(" ", "_"), parent_id=root_folder)
    date_folder = get_or_create_folder(service, str(date_input), parent_id=class_folder)

    # --- Upload File ---
    file_buffer = io.BytesIO(uploaded_file.getvalue())
    uploaded_id = upload_file_to_drive(service, file_buffer, file_name, parent_folder_id=date_folder)

    st.success("✅ Uploaded to Google Drive!")
    st.markdown(f"📄 File ID: `{uploaded_id}`")
    st.balloons()
