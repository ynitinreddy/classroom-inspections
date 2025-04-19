# 🧠 AI Classroom Inspector (ASU Edition)

An AI-powered classroom inspection tool that leverages **YOLOv8** for object detection and **OpenAI's GPT-4 Vision** for intelligent analysis of classroom images. Designed to streamline classroom maintenance reporting with automated visual reasoning, anomaly detection, and DOCX report generation — all through a clean and intuitive **Streamlit** interface.

## 🚀 Features

- 📷 **Image Upload & Preview** – Upload classroom images from various angles  
- 🧠 **Smart Visual Inspection** – Uses GPT-4 Vision to generate inspection reports across 13 classroom aspects  
- 🔍 **YOLOv8 Anomaly Detection** – Optional object detection for flagging unusual items (e.g., trash, food, phones)  
- 📝 **Automated DOCX Report** – Generates formatted, downloadable reports for recordkeeping or auditing  
- ⚙️ **Custom Prompts & Model Toggling** – Choose between GPT models and customize inspection logic  
- 💡 **Built for ASU** – Branded with ASU colors, layout, and tailored classroom prompts

## 🖼️ Screenshots

<!-- You can update these once deployed -->
<p align="center">
  <img src="screenshots/upload_page.png" width="700" />
  <img src="screenshots/anomaly_detection.png" width="700" />
  <img src="screenshots/report_preview.png" width="700" />
</p>

## 🧩 Tech Stack

- **Frontend/UI**: Streamlit  
- **Object Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- **LLM & Vision**: OpenAI GPT-4 / GPT-4o (via API)  
- **Image Handling**: PIL, NumPy  
- **Report Generation**: python-docx  
- **Deployment Ready**: Supports local dev and Streamlit Cloud secrets management

---

## ⚙️ Local Setup

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/ai-classroom-inspector.git
cd ai-classroom-inspector
```
