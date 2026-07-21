# UAV Disaster Detection & Alerting System

A real-time disaster monitoring platform that uses a UAV (drone) mounted camera, YOLOv8 object detection, and a Flask web dashboard to detect humans in disaster zones and trigger automated SMS alerts to emergency contacts.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [Security Notes](#security-notes)
- [License](#license)

---

## Overview

The **UAV Disaster Detection & Alerting System** is designed for emergency response scenarios. A drone streams live video footage which is processed by a YOLOv8 model to detect the presence of humans. When humans are detected, the system:

- Determines a **severity level** (NONE / MEDIUM / HIGH)
- Logs a **GPS-tagged alert** with live weather data
- Dispatches **SMS notifications** to registered emergency contacts via Twilio
- Displays everything on a **real-time web dashboard**

---

## Features

- 🎯 **Real-time Human Detection** — YOLOv8 nano model runs inference on live camera or captured phone images
- 📡 **Live Video Stream** — MJPEG feed served directly from the Flask backend
- 🗺️ **GPS + Weather Integration** — Simulated UAV GPS coordinates enriched with Open-Meteo weather data and reverse-geocoded via Nominatim
- 📱 **Automated SMS Alerts** — Twilio integration sends role-based emergency notifications
- 👥 **Contact Management** — Add, edit, and delete emergency contacts with roles (Admin / Responder / Observer)
- 🔐 **User Authentication** — Local username/password login + Google OAuth 2.0
- 📊 **Alert Dashboard** — Live feed of detection events, human count, severity, and message logs
- 📸 **Image Capture Logging** — Detected frames saved automatically to `phone_captured_images/`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.x, Flask |
| Object Detection | Ultralytics YOLOv8 (`yolov8n.pt`) |
| Computer Vision | OpenCV |
| SMS | Twilio REST API |
| Auth | Session-based + Google OAuth 2.0 |
| Database | SQLite (`uav_contacts.db`) |
| Weather | Open-Meteo API |
| Geocoding | Nominatim (OpenStreetMap) |
| Frontend | HTML/CSS/JS (Jinja2 templates) |

---

## Project Structure

```
Program/
├── app.py                      # Main Flask application
├── test_sms.py                 # Standalone SMS test script
├── .env                        # Environment variables (secrets)
├── .gitignore
├── yolov8n.pt                  # YOLOv8 nano model weights
├── uav_contacts.db             # SQLite database (auto-created)
├── phone_captured_images/      # Auto-saved detection frames
├── static/
│   ├── drone.png
│   └── img/
│       ├── drone_bg.png
│       ├── emergencies_bg.jpg / .png
│       └── login_bg.png
├── templates/
│   ├── index.html              # Login page
│   ├── login.html              # Auth page
│   └── dashboard.html          # Main monitoring dashboard
├── UAV-DISASTER-DETECTION/     # Git submodule / reference repo
└── venv/                       # Python virtual environment
```

---

## Prerequisites

- Python 3.10+
- A webcam or IP camera (or pre-captured images in `phone_captured_images/`)
- [Twilio account](https://www.twilio.com/) with a phone number
- [Google Cloud Console](https://console.cloud.google.com/) project with OAuth 2.0 credentials (optional)
- Internet connection (for weather and geocoding APIs)

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/UAV-DISASTER-DETECTION.git
cd UAV-DISASTER-DETECTION/Program
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install flask ultralytics opencv-python twilio python-dotenv requests
```

---

## Configuration

Create a `.env` file in the `Program/` directory (or edit the existing one):

```env
# Twilio SMS Configuration
TWILIO_SID=your_twilio_account_sid
TWILIO_TOKEN=your_twilio_auth_token
TWILIO_FROM=+1XXXXXXXXXX

# Flask App Secret
APP_SECRET_KEY=your_random_secret_key

# Google OAuth (optional)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=http://localhost:5000/google_callback
```

> ⚠️ **Never commit your `.env` file to version control.** It is already listed in `.gitignore`.

You can also adjust these constants at the top of `app.py`:

```python
CAMERA_SOURCE    = 0          # 0 = default webcam, or RTSP URL
CAPTURE_INTERVAL = 10         # seconds between captures
YOLO_EVERY       = 3          # run YOLO on every Nth frame
```

---

## Running the Application

```bash
cd Program
python app.py
```

Then open your browser at: **http://localhost:5000**

---

## Usage

1. **Register / Log in** using the login page (local account or Google)
2. Navigate to the **Dashboard**
3. Click **Start Camera** to begin the live UAV feed
4. The system automatically runs YOLOv8 detection and logs alerts
5. Use the **Contacts** panel to add emergency personnel with their phone numbers and roles
6. Alerts with detected human counts trigger automatic SMS dispatches to registered contacts
7. View the **Alert Log** and **Message Log** panels for a history of events

---

## How It Works

```
Camera Feed (webcam / drone)
        │
        ▼
  OpenCV Frame Capture
        │
        ▼
  YOLOv8 Inference (every Nth frame)
        │
        ├── Humans Detected? ──► Calculate Severity (MEDIUM / HIGH)
        │                               │
        │                               ▼
        │                     GPS + Weather Fetch
        │                               │
        │                               ▼
        │                     POST /alert → Update State
        │                               │
        │                               ▼
        │                     Send SMS via Twilio
        │
        ▼
  MJPEG Stream → Dashboard (/video_feed)
        │
        ▼
  Real-time Dashboard (human count, severity, GPS, alert log)
```

---

## Security Notes

- Rotate your Twilio credentials and Google OAuth secrets before any public deployment
- The `.env` file contains sensitive credentials — keep it out of version control
- The default `APP_SECRET_KEY` should be replaced with a long random string in production
- For production deployment, run behind a reverse proxy (nginx) with HTTPS

---

## License

This project is developed for academic and emergency response research purposes.
