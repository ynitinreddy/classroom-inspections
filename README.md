# 🧠 AI-Powered Classroom Inspector (ASU Edition)

This is a Streamlit-based web application that uses computer vision and GPT-powered AI to automatically inspect classroom environments based on uploaded images. It was developed to help automate and streamline classroom condition reporting at Arizona State University (ASU).

The app supports:
- 🖼️ Visual inspection using GPT-4o Vision
- 🚨 Anomaly detection with YOLOv8
- 📄 Automated report generation (Word & TXT formats)

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ynitinreddy/classroom-inspections.git
cd classroom-inspector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API keys

#### • Local development:
Create a `.env` file or modify the path in code:

```env
OPENAI_API_KEY=your_openai_key
```

#### • Streamlit Cloud:
Use the built-in secrets manager. In `.streamlit/secrets.toml`:

```bash
[secrets]
OPENAI_API_KEY = "your_openai_key"
```

### 4. Run the app

```bash
streamlit run app.py
```


## ✨ Features & App Flow

1. **Upload Classroom Images**
   - Accepts JPG, JPEG, or PNG files.
   - Multiple angles recommended for best results.

2. **(Optional) Add Classroom Number**
   - This appears in the final report and filename.

3. **Choose Model Type**
   - 🔹 Best (GPT-4o): Fast, accurate, low-cost
   - 🔹 Basic (GPT-4o-mini): Light and fast
   - 🔹 Expert: Placeholder for advanced reasoning (can be configured)

4. **Enable Anomaly Detection (YOLOv8)**
   - Identifies items like trash, backpacks, utensils, etc.
   - Only runs if checkbox is selected.

5. **Inspection Prompt**
   - Uses a 13-point checklist prompt.
   - Fully customizable via an editable text area.

6. **Run Inspection**
   - Calls GPT Vision API using uploaded images (and anomaly data if enabled).
   - Returns a short, structured classroom report.

7. **Generate Report**
   - Automatically builds a downloadable `.docx` and `.txt` report.
   - Includes original and annotated images (if anomalies were found).

---

## 🏗️ Tech Stack & Architecture

### 🐍 Python Libraries
- **Streamlit** – UI & interactivity
- **Pillow (PIL)** – Image manipulation
- **NumPy** – Array handling for image data
- **OpenAI** – GPT-4o Vision model access
- **Ultralytics YOLO** – Object detection (YOLOv8)
- **python-docx** – DOCX report generation
- **dotenv** – Environment variable loading

### 🧠 AI & CV Models
- **GPT-4o / GPT-4o-mini** – For image captioning, checklist reasoning
- **YOLOv8 (yolov8n.pt)** – Lightweight anomaly detector (runs locally)

### 💾 File Handling
- Uploaded images and model-generated results are kept in memory during session.
- Final report is generated as a `BytesIO` object for direct download.

### 📄 Output
- `classroom_inspection_report.docx`
- `inspection_report.txt`

---


## 💡 Usage Tips & Notes

- 🔄 **Clearing Uploads:**  
  If you want to re-upload different images, use the "🗑️ Clear Uploaded Images" button. This resets the uploader widget.

- 🧠 **Prompt Customization:**  
  You can tweak the inspection prompt using the "⚙️ More Options" expander to fine-tune how GPT interprets the classroom images.

- 📸 **Multiple Angles Recommended:**  
  Upload images from different corners of the room for better inspection accuracy.

- ⚠️ **Anomaly Detection is Optional:**  
  YOLOv8 runs only if the checkbox is selected — it adds extra visual feedback by highlighting unusual objects (like trash, backpacks, food, etc.).

- 📝 **Report Downloads:**  
  After the inspection is run, you can download a full `.docx` report or a lightweight `.txt` version directly.

- 🧪 **Model Modes:**
  - *Best:* GPT-4o, optimized for quality & cost.
  - *Basic:* Lighter, faster GPT-4o-mini.
  - *Expert:* Placeholder – to be extended with advanced capabilities.

> 📍 **Reminder:** The AI is vision-based — only visible issues are evaluated.

## 🚧 Limitations & Future Improvements

### Known Limitations:
- 🔍 **Visibility Dependent:**
  The AI can only inspect what's clearly visible in uploaded images. Obstructed or poorly lit areas may result in "Cannot determine" responses.

- 📦 **No Persistent Storage:**
  Uploaded files and generated reports exist only during the Streamlit session.

- 📶 **Requires Internet:**
  GPT model access and YOLOv8 download require an internet connection.

- 🧠 **Expert Model Placeholder:**
  The “Expert” model option is not fully configured yet. Currently defaults to GPT-4o.

- 🤖 **YOLO Model Generalization:**
  YOLOv8 (yolov8n.pt) may detect extra or irrelevant objects if classroom layouts vary widely.

---

### Potential Enhancements:
- ✅ Add persistent file storage or cloud integration (e.g., S3, Google Drive).
- 🧩 Improve YOLO anomaly list or use a custom-trained model.
- 🧠 Enable true “Expert” model (e.g., GPT-4 Turbo with fine-tuned prompts).
- 🎨 Add image preview with annotations side-by-side.
- 🔒 Role-based access for managing inspections per user or department.




