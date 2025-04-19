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
Use the built-in secrets manager. In .streamlit/secrets.toml:

