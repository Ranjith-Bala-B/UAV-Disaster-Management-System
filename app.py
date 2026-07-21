import sys
import cv2
import os
import time
import random
import threading
import requests
import hashlib
import sqlite3
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for, session, send_from_directory
from functools import wraps
from ultralytics import YOLO

# Add local libs to sys.path to resolve deep-path loading issues on Windows
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs'))

from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv

load_dotenv()

CAMERA_SOURCE    = 0
CAMERA_MODE      = os.getenv("CAMERA_MODE", "local").strip().lower()
IMAGE_FOLDER     = os.getenv("IMAGE_FOLDER", "phone_captured_images").strip()
ALERT_SERVER_URL = "http://localhost:5000/alert"
CAPTURE_INTERVAL = 10
YOLO_EVERY       = 3

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///uav_contacts.db")

def _resolve_db_path(database_url):
    """Convert a sqlite:/// style URL into a filesystem path for sqlite3.connect()."""
    if database_url.startswith("sqlite:///"):
        return database_url.replace("sqlite:///", "", 1) or "uav_contacts.db"
    return database_url

DB_PATH = _resolve_db_path(DATABASE_URL)

# Ensure parent directory of SQLite database exists
_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)

TWILIO_SID        = os.getenv("TWILIO_SID")
TWILIO_TOKEN      = os.getenv("TWILIO_TOKEN")
TWILIO_FROM       = os.getenv("TWILIO_FROM")
GOOGLE_CLIENT_ID  = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5000/google_callback")

os.makedirs(IMAGE_FOLDER, exist_ok=True)

app            = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY", "fallback_secret_key")
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
model          = YOLO("yolov8n.pt")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT NOT NULL UNIQUE,
                password   TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = [row["name"] for row in cursor.fetchall()]
        new_cols = {
            "name": "TEXT DEFAULT ''",
            "email": "TEXT DEFAULT ''",
            "phone": "TEXT DEFAULT ''",
            "role": "TEXT DEFAULT 'common user'",
            "designation": "TEXT DEFAULT ''",
            "address": "TEXT DEFAULT ''"
        }
        for col_name, col_type in new_cols.items():
            if col_name not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN {} {}".format(col_name, col_type))

        conn.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                phone      TEXT NOT NULL,
                role       TEXT NOT NULL,
                address    TEXT DEFAULT '',
                notes      TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        
        # Add email column to contacts if missing
        cursor = conn.execute("PRAGMA table_info(contacts)")
        c_columns = [row["name"] for row in cursor.fetchall()]
        if "email" not in c_columns:
            conn.execute("ALTER TABLE contacts ADD COLUMN email TEXT DEFAULT ''")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                sent_to   TEXT NOT NULL,
                phone     TEXT NOT NULL,
                role      TEXT NOT NULL,
                message   TEXT NOT NULL,
                status    TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT NOT NULL,
                disaster_type  TEXT,
                humans         INTEGER,
                severity       TEXT,
                latitude       REAL,
                longitude      REAL,
                altitude       REAL,
                location       TEXT,
                image          TEXT
            )
        """)
        
        # Add temp, wind, weather_code columns to alerts if missing
        cursor = conn.execute("PRAGMA table_info(alerts)")
        a_columns = [row["name"] for row in cursor.fetchall()]
        if "temp" not in a_columns:
            conn.execute("ALTER TABLE alerts ADD COLUMN temp TEXT DEFAULT '--'")
        if "wind" not in a_columns:
            conn.execute("ALTER TABLE alerts ADD COLUMN wind TEXT DEFAULT '--'")
        if "weather_code" not in a_columns:
            conn.execute("ALTER TABLE alerts ADD COLUMN weather_code TEXT DEFAULT '--'")

        conn.commit()
    print("Database initialized: {}".format(DB_PATH))

init_db()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in") or not session.get("username"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


state = {
    "human_count":      0,
    "severity":         "NONE",
    "gps":              {"latitude": 0, "longitude": 0, "altitude": 0, "weather": {"temp": "--", "wind": "--", "code": "--"}},
    "last_alert_time":  "-",
    "alerts":           [],
    "total_detections": 0,
    "messages":         [],
}
lock          = threading.Lock()
raw_frame     = None
output_frame  = None
camera_active = False
last_browser_gps = None


twilio_client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)

def send_sms(phone, message):
    number = phone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    e164 = None
    if number.startswith("+"):
        e164 = number
    elif number.startswith("91") and len(number) == 12 and number.isdigit():
        e164 = "+" + number
    elif number.startswith("0") and len(number) == 11:
        e164 = "+91" + number[1:]
    elif len(number) == 10 and number.isdigit():
        e164 = "+91" + number
    else:
        return False, "Invalid phone number: " + phone

    print("Sending SMS to: {}".format(e164))
    try:
        msg = twilio_client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=e164
        )
        print("SMS sent to {} SID: {}".format(e164, msg.sid))
        return True, "Sent"
    except TwilioRestException as e:
        print("Twilio error for {}: {}".format(e164, e.msg))
        return False, e.msg
    except Exception as e:
        print("SMS error for {}: {}".format(e164, e))
        return False, str(e)


def send_whatsapp(phone, message, media_url=None):
    number = phone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    e164 = None
    if number.startswith("+"):
        e164 = number
    elif number.startswith("91") and len(number) == 12 and number.isdigit():
        e164 = "+" + number
    elif number.startswith("0") and len(number) == 11:
        e164 = "+91" + number[1:]
    elif len(number) == 10 and number.isdigit():
        e164 = "+91" + number
    else:
        return False, "Invalid phone number: " + phone

    to_whatsapp = "whatsapp:{}".format(e164)
    from_whatsapp = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    
    print("Sending WhatsApp message from {} to {}".format(from_whatsapp, to_whatsapp))
    try:
        kwargs = {
            "body": message,
            "from_": from_whatsapp,
            "to": to_whatsapp
        }
        if media_url:
            kwargs["media_url"] = [media_url]
            
        msg = twilio_client.messages.create(**kwargs)
        print("WhatsApp sent to {} SID: {}".format(to_whatsapp, msg.sid))
        return True, "Sent"
    except TwilioRestException as e:
        print("Twilio WhatsApp error for {}: {}".format(to_whatsapp, e.msg))
        return False, e.msg
    except Exception as e:
        print("WhatsApp error for {}: {}".format(to_whatsapp, e))
        return False, str(e)


def trigger_whatsapp_alerts(humans, gps, image_path, location_name):
    # Formulate message
    message_body = (
        "🚨 *UAV DISASTER ALERT: HUMAN DETECTED* 🚨\n\n"
        "👤 *Humans Count:* {}\n"
        "⚠️ *Severity:* {}\n"
        "📍 *Location:* {}\n"
        "🌐 *Coordinates:* {:.6f}, {:.6f}\n"
        "⏰ *Timestamp:* {}\n\n"
        "🗺️ *Google Maps Link:* https://www.google.com/maps?q={},{}"
    ).format(
        humans,
        severity_level(humans),
        location_name,
        gps["latitude"],
        gps["longitude"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        gps["latitude"],
        gps["longitude"]
    )
    
    # Formulate media URL
    media_url = None
    if image_path:
        public_url = os.getenv("PUBLIC_SERVER_URL")
        if public_url:
            clean_url = public_url.strip().rstrip("/")
            media_url = "{}/{}".format(clean_url, image_path)
        else:
            media_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"

    # Verify reachability of the generated media URL to avoid Twilio failing delivery asynchronously (Error 63019)
    if media_url:
        try:
            resp = requests.get(media_url, stream=True, timeout=1.5)
            if resp.status_code != 200:
                print("Generated media URL {} is not reachable (status {}). Twilio would fail to deliver this message. Removing media URL.".format(media_url, resp.status_code))
                media_url = None
            else:
                print("Generated media URL {} is reachable. Sending with media.".format(media_url))
        except Exception as e:
            print("Generated media URL {} check failed ({}). Twilio would fail to deliver this message. Removing media URL.".format(media_url, e))
            media_url = None

    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM contacts WHERE LOWER(role) IN ('rescue team', 'rescue member', 'uav community', 'community member', 'volunteer') OR LOWER(role) LIKE '%uav%' OR LOWER(role) LIKE '%community%' OR LOWER(role) LIKE '%rescue%'"
            ).fetchall()
            contacts = [dict(r) for r in rows]
    except Exception as e:
        print("Database error in trigger_whatsapp_alerts:", e)
        contacts = []
        
    if not contacts:
        print("No rescue contacts found in database to send WhatsApp alert.")
        return
        

    for contact in contacts:
        ok, status_msg = send_whatsapp(contact["phone"], message_body, media_url)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_text = "WhatsApp Sent" if ok else "WhatsApp Failed: {}".format(status_msg)
        try:
            with get_db() as db:
                db.execute(
                    "INSERT INTO message_log (sent_to,phone,role,message,status,timestamp) VALUES (?,?,?,?,?,?)",
                    (contact["name"], contact["phone"], contact["role"], "[WhatsApp Alert] {}".format(location_name), status_text, ts)
                )
                db.commit()
            with lock:
                state["messages"].insert(0, {
                    "timestamp": ts, 
                    "to": contact["name"], 
                    "phone": contact["phone"],
                    "role": contact["role"], 
                    "message": "[WhatsApp Alert] {}".format(location_name), 
                    "status": status_text,
                })
        except Exception as e:
            print("Failed to log WhatsApp alert status to DB:", e)


def get_gps():
    global last_browser_gps
    with lock:
        if last_browser_gps is not None:
            lat = last_browser_gps["latitude"]
            lon = last_browser_gps["longitude"]
            alt = last_browser_gps["altitude"] if last_browser_gps["altitude"] > 0 else random.randint(30, 80)
        else:
            lat = round(random.uniform(12.95, 13.05), 6)
            lon = round(random.uniform(77.55, 77.65), 6)
            alt = random.randint(30, 80)
    weather = {"temp": "--", "wind": "--", "code": "--"}
    try:
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        w_data = requests.get(w_url, timeout=2).json()
        if "current_weather" in w_data:
            weather = {
                "temp": w_data["current_weather"].get("temperature", "--"),
                "wind": w_data["current_weather"].get("windspeed", "--"),
                "code": w_data["current_weather"].get("weathercode", "--")
            }
    except Exception:
        pass
    return {
        "latitude":  lat,
        "longitude": lon,
        "altitude":  alt,
        "weather":   weather
    }


def reverse_geocode(lat, lon):
    try:
        headers = {"User-Agent": "UAVDisasterIdentificationAlertingSystem/1.0 (ranji@orionforge.com)"}
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=16"
        resp = requests.get(url, headers=headers, timeout=2.5)
        if resp.status_code == 200:
            res_json = resp.json()
            addr = res_json.get("address", {})
            road = addr.get("road")
            suburb = addr.get("suburb") or addr.get("neighbourhood") or addr.get("subdivision") or addr.get("residential")
            city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county")
            state = addr.get("state")
            
            elements = [e for e in [road, suburb, city, state] if e]
            if elements:
                return ", ".join(elements)
            return res_json.get("display_name", f"Sector near {lat:.4f}, {lon:.4f}")
    except Exception as e:
        print("Nominatim geocoding error:", e)
        

    fallbacks = [
        "Koramangala 5th Block, Bengaluru, Karnataka",
        "Indiranagar 12th Main Rd, Bengaluru, Karnataka",
        "M.G. Road Metro Area, Bengaluru, Karnataka",
        "Jayanagar 4th T Block, Bengaluru, Karnataka",
        "Hebbal Outer Ring Rd, Bengaluru, Karnataka",
        "Whitefield ITPL Main Rd, Bengaluru, Karnataka",
        "HSR Layout Sector 2, Bengaluru, Karnataka",
        "Malleshwaram Margosa Road, Bengaluru, Karnataka",
        "Electronic City Phase 1 Tollway, Bengaluru, Karnataka",
        "Sadashivanagar Near Sankey Tank, Bengaluru, Karnataka"
    ]

    coord_val = int((abs(lat) * 1000) + (abs(lon) * 1000))
    idx = coord_val % len(fallbacks)
    return fallbacks[idx]


def severity_level(humans):
    if humans == 0: return "NONE"
    if humans == 1: return "MEDIUM"
    return "HIGH"

def db_row_to_alert_dict(row):
    lat = row["latitude"]
    lon = row["longitude"]
    alt = row["altitude"]
    loc = row["location"]
    # Check keys safely
    row_keys = row.keys()
    temp = row["temp"] if "temp" in row_keys else "--"
    wind = row["wind"] if "wind" in row_keys else "--"
    weather_code = row["weather_code"] if "weather_code" in row_keys else "--"
    
    return {
        "id":            row["id"],
        "timestamp":     row["timestamp"],
        "disaster_type": row["disaster_type"],
        "humans":        row["humans"],
        "humans_detected": row["humans"],
        "severity":      row["severity"],
        "location":      loc,
        "image":         row["image"],
        "temp":          temp,
        "wind":          wind,
        "weather_code":  weather_code,
        "gps": {
            "latitude":  lat if lat is not None else 0.0,
            "longitude": lon if lon is not None else 0.0,
            "altitude":  alt if alt is not None else 0.0,
            "location":  loc if loc is not None else ""
        },
        "gps_location": {
            "latitude":  lat if lat is not None else 0.0,
            "longitude": lon if lon is not None else 0.0,
            "altitude":  alt if alt is not None else 0.0,
            "location":  loc if loc is not None else ""
        }
    }

def find_duplicate_alert(lat, lon, current_time_str):
    if lat is None or lon is None:
        return None
    try:
        current_dt = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        current_dt = datetime.now()
        
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 50").fetchall()
        for row in rows:
            try:
                alert_dt = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                diff = abs((current_dt - alert_dt).total_seconds())
                if diff <= 300: # 5 minutes
                    a_lat = row["latitude"]
                    a_lon = row["longitude"]
                    if a_lat is not None and a_lon is not None:
                        if abs(a_lat - lat) < 0.001 and abs(a_lon - lon) < 0.001:
                            return row
            except Exception as e:
                print("Error parsing timestamp in duplicate check:", e)
    return None



def add_or_merge_alert(alert):
    gps = alert.get("gps") or alert.get("gps_location")
    try:
        lat = gps.get("latitude") if gps else None
        lon = gps.get("longitude") if gps else None
        alt = gps.get("altitude") if gps else None
        loc = alert.get("location") or (gps.get("location") if gps else "") or ""
        humans = alert.get("humans") or alert.get("humans_detected") or 0
        timestamp = alert.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        weather = gps.get("weather", {}) if gps else {}
        temp = str(weather.get("temp", "--"))
        wind = str(weather.get("wind", "--"))
        weather_code = str(weather.get("code", "--"))
        
        duplicate = find_duplicate_alert(lat, lon, timestamp)
        if duplicate:
            dup_id = duplicate["id"]
            merged_humans = max(duplicate["humans"] or 0, humans)
            severity = severity_level(merged_humans)
            disaster_type = alert.get("disaster_type") or duplicate["disaster_type"] or "Human Presence Detected - Natural Disaster Zone"
            image = alert.get("image") or duplicate["image"] or ""
            
            with get_db() as conn:
                conn.execute(
                    "UPDATE alerts SET timestamp=?, disaster_type=?, humans=?, severity=?, image=?, temp=?, wind=?, weather_code=? WHERE id=?",
                    (timestamp, disaster_type, merged_humans, severity, image, temp, wind, weather_code, dup_id)
                )
                conn.commit()
            
            with lock:
                state["alerts"] = [a for a in state["alerts"] if a.get("id") != dup_id]
                
            alert["id"] = dup_id
            alert["humans"] = merged_humans
            alert["humans_detected"] = merged_humans
            alert["severity"] = severity
            alert["image"] = image
            alert["timestamp"] = timestamp
            alert["temp"] = temp
            alert["wind"] = wind
            alert["weather_code"] = weather_code
            
            with lock:
                state["alerts"].insert(0, alert)
                state["alerts"] = state["alerts"][:50]
                
            print("Alert merged successfully. ID:", dup_id)
            return False # Merged, not new
        else:
            severity = alert.get("severity") or severity_level(humans)
            disaster_type = alert.get("disaster_type") or "Human Presence Detected - Natural Disaster Zone"
            image = alert.get("image") or ""
            
            with get_db() as conn:
                cur = conn.execute(
                    "INSERT INTO alerts (timestamp, disaster_type, humans, severity, latitude, longitude, altitude, location, image, temp, wind, weather_code) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (timestamp, disaster_type, humans, severity, lat, lon, alt, loc, image, temp, wind, weather_code)
                )
                alert["id"] = cur.lastrowid
                conn.commit()
                
            alert["temp"] = temp
            alert["wind"] = wind
            alert["weather_code"] = weather_code
            
            with lock:
                state["alerts"].insert(0, alert)
                state["alerts"] = state["alerts"][:50]
            return True # New alert created
    except Exception as e:
        print("Database error in add_or_merge_alert:", e)
        return True

def send_alert(humans, gps):
    sev = severity_level(humans)
    alert_data = {
        "humans_detected": humans,
        "severity":        sev,
        "gps_location":    gps,
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("ALERT:", alert_data)
    try:
        requests.post(ALERT_SERVER_URL, json=alert_data, timeout=1)
    except Exception:
        print("Alert server not reachable")


def capture_loop():
    global raw_frame, camera_active
    print("UAV Disaster Monitoring System Started (Mode: {})".format(CAMERA_MODE.upper()))
    
    if CAMERA_MODE == "browser":
        while True:
            if not camera_active:
                with lock:
                    raw_frame = None
            time.sleep(0.5)
        return

    cap = None
    while True:
        if not camera_active:
            if cap is not None:
                cap.release()
                cap = None
                with lock:
                    raw_frame = None
            time.sleep(0.5)
            continue
        if cap is None:
            cap = cv2.VideoCapture(CAMERA_SOURCE)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print("Unable to open laptop camera - retrying...")
                cap = None
                time.sleep(1)
                continue
            print("Laptop camera connected")
        ret, frame = cap.read()
        if not ret:
            print("Frame not received - reconnecting...")
            cap.release()
            cap = None
            time.sleep(0.5)
            continue
        frame = cv2.resize(frame, (640, 480))
        with lock:
            raw_frame = frame


def detection_loop():
    global output_frame
    last_capture     = time.time()
    frame_idx        = 0
    last_boxes       = []
    last_human_count = 0
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        print("Warning: Failed to load face cascade classifier:", e)
        face_cascade = None

    while True:
        with lock:
            frame = raw_frame.copy() if raw_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue

        frame_idx += 1

        if frame_idx % YOLO_EVERY == 0:
            results = model(frame, verbose=False, imgsz=320)
            last_boxes       = []
            last_human_count = 0
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        last_human_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        last_boxes.append((x1, y1, x2, y2, conf))

        # Copy frame for clear saving (unblurred faces, but has annotations)
        clear_frame = frame.copy()

        # Perform face detection and blur on main frame for live feed and default saved image
        if last_human_count > 0 and face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(10, 10))
                for (fx, fy, fw, fh) in faces:
                    h_img, w_img = frame.shape[:2]
                    fy_start = max(0, fy)
                    fy_end = min(h_img, fy + fh)
                    fx_start = max(0, fx)
                    fx_end = min(w_img, fx + fw)
                    if fy_end > fy_start and fx_end > fx_start:
                        face_roi = frame[fy_start:fy_end, fx_start:fx_end]
                        if face_roi.size > 0:
                            # Apply strong GaussianBlur
                            blurred_roi = cv2.GaussianBlur(face_roi, (51, 51), 0)
                            frame[fy_start:fy_end, fx_start:fx_end] = blurred_roi
            except Exception as e:
                print("Face blur error:", e)

        # Annotate main frame (blurred faces)
        for (x1, y1, x2, y2, conf) in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)
            cv2.putText(frame, "Human {:.0%}".format(conf), (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)

        sev   = severity_level(last_human_count)
        color = (0, 255, 80) if sev == "NONE" else (0, 165, 255) if sev == "MEDIUM" else (0, 0, 255)
        cv2.putText(frame, "Humans: {}  [{}]".format(last_human_count, sev), (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (15, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Annotate clear frame (no face blur)
        for (x1, y1, x2, y2, conf) in last_boxes:
            cv2.rectangle(clear_frame, (x1, y1), (x2, y2), (0, 255, 80), 2)
            cv2.putText(clear_frame, "Human {:.0%}".format(conf), (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)
        cv2.putText(clear_frame, "Humans: {}  [{}]".format(last_human_count, sev), (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(clear_frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (15, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        now = time.time()
        if last_human_count > 0 and (now - last_capture >= CAPTURE_INTERVAL):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = "{}/detection_{}.jpg".format(IMAGE_FOLDER, ts)
            clear_path = "{}/detection_{}_clear.jpg".format(IMAGE_FOLDER, ts)
            
            # Save both blurred and clear versions
            cv2.imwrite(path, frame)
            cv2.imwrite(clear_path, clear_frame)
            print("Images saved: {} (blurred) and {} (clear)".format(path, clear_path))
            
            gps = get_gps()
            location_name = reverse_geocode(gps["latitude"], gps["longitude"])
            gps["location"] = location_name

            alert = {
                "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "disaster_type": "Human Presence Detected - Natural Disaster Zone",
                "humans":       last_human_count,
                "severity":     sev,
                "gps":          gps,
                "location":     location_name,
                "image":        path,
            }
            with lock:
                state["human_count"]      = last_human_count
                state["severity"]         = sev
                state["disaster_type"]    = alert["disaster_type"]
                state["gps"]              = gps
                state["last_alert_time"]  = alert["timestamp"]
                state["total_detections"] += last_human_count
            
            # Merges alert or adds a new one. Returns True if a new alert is created
            is_new = add_or_merge_alert(alert)
            
            if is_new:
                threading.Thread(
                    target=trigger_whatsapp_alerts, 
                    args=(last_human_count, gps, path, location_name), 
                    daemon=True
                ).start()

            
            last_capture = now

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with lock:
            output_frame = buf.tobytes()


def gen_frames():
    global output_frame
    while True:
        with lock:
            frame = output_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.02)



@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (username, hash_password(password))
            ).fetchone()
        if user:
            global camera_active
            camera_active       = True
            session.permanent   = True
            session["logged_in"] = True
            session["username"]  = username
            return redirect(url_for("home"))
        error = "Invalid username or password."
    return render_template("login.html", error=error, google_client_id=GOOGLE_CLIENT_ID)


@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    confirm  = request.form.get("confirm",  "").strip()
    if not username or not password:
        return render_template("login.html", signup_error="Username and password are required.", show_signup=True, google_client_id=GOOGLE_CLIENT_ID)
    if password != confirm:
        return render_template("login.html", signup_error="Passwords do not match.", show_signup=True, google_client_id=GOOGLE_CLIENT_ID)
    if len(password) < 6:
        return render_template("login.html", signup_error="Password must be at least 6 characters.", show_signup=True, google_client_id=GOOGLE_CLIENT_ID)
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username,password,created_at) VALUES (?,?,?)",
                (username, hash_password(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
        global camera_active
        camera_active        = True
        session.clear()
        session["logged_in"] = True
        session["username"]  = username
        session.modified     = True
        return redirect(url_for("home"))
    except sqlite3.IntegrityError:
        return render_template("login.html", signup_error="Username already exists.", show_signup=True, google_client_id=GOOGLE_CLIENT_ID)


@app.route("/google_callback")
def google_callback():
    code = request.args.get("code")
    if not code:
        return redirect(url_for("login"))
    try:
        import urllib.request, urllib.parse, json as _json
        GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
        redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5000/google_callback")
        token_data = urllib.parse.urlencode({
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code"
        }).encode()
        token_req = urllib.request.Request("https://oauth2.googleapis.com/token", data=token_data, method="POST")
        token_req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(token_req) as resp:
            token_json = _json.loads(resp.read())
        access_token = token_json.get("access_token", "")
        user_req = urllib.request.Request(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": "Bearer " + access_token}
        )
        with urllib.request.urlopen(user_req) as resp:
            user_info = _json.loads(resp.read())
        email    = user_info.get("email", "")
        name     = user_info.get("name", email.split("@")[0])
        username = name.replace(" ", "_").lower()
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            if not user:
                conn.execute(
                    "INSERT INTO users (username,password,created_at) VALUES (?,?,?)",
                    (username, hash_password(email + "_google"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                conn.commit()
        global camera_active
        camera_active        = True
        session["logged_in"] = True
        session["username"]  = username
        return redirect(url_for("home"))
    except Exception as e:
        return redirect(url_for("login"))

@app.route("/logout")
def logout():
    global camera_active
    camera_active = False
    session.clear()
    return redirect(url_for("login"))


@app.route("/profile", methods=["GET"])
@login_required
def get_profile():
    username = session.get("username")
    with get_db() as conn:
        user = conn.execute(
            "SELECT username, name, email, phone, role, designation, address FROM users WHERE username=?",
            (username,)
        ).fetchone()
    if user:
        return jsonify(dict(user))
    return jsonify({"error": "User not found"}), 404


@app.route("/profile", methods=["POST"])
@login_required
def update_profile():
    data = request.json or request.form
    username = session.get("username")
    
    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()
    role = data.get("role", "common user").strip()
    designation = data.get("designation", "").strip()
    address = data.get("address", "").strip()
    
    valid_roles = ["common user", "volunteer", "victim", "rescue member"]
    if role not in valid_roles:
        role = "common user"
        
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET name=?, email=?, phone=?, role=?, designation=?, address=? WHERE username=?",
            (name, email, phone, role, designation, address, username)
        )
        conn.commit()
        
    return jsonify({"status": "success", "message": "Profile updated successfully"})


@app.route("/")
@login_required
def home():
    return render_template("index.html", username=session.get("username"))


@app.route("/dashboard")
@login_required
def dashboard():
    global camera_active
    camera_active = True
    return render_template("dashboard.html", username=session.get("username"))


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/phone_captured_images/<path:filename>")
def serve_detection_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


@app.route("/alerts/<int:alert_id>/pdf")
@login_required
def generate_alert_pdf(alert_id):
    # Fetch alert from DB
    with get_db() as conn:
        row = conn.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
    if not row:
        return "Alert not found", 404
        
    alert = db_row_to_alert_dict(row)
    
    # Safely get weather metrics
    row_keys = row.keys()
    temp = row["temp"] if "temp" in row_keys else "--"
    wind = row["wind"] if "wind" in row_keys else "--"
    weather_code = row["weather_code"] if "weather_code" in row_keys else "--"
    
    # Fetch rescue/volunteer contacts
    with get_db() as conn:
        contacts_rows = conn.execute(
            "SELECT * FROM contacts WHERE LOWER(role) IN ('rescue team', 'rescue member', 'volunteer') ORDER BY role, name"
        ).fetchall()
        response_team = [dict(r) for r in contacts_rows]
        
    # Fetch Timeline: messages sent around this alert
    timeline_events = []
    timeline_events.append({
        "time": alert["timestamp"],
        "event": "UAV Identification",
        "details": f"Disaster identified: {alert.get('disaster_type')}. {alert['humans']} human(s) detected. Severity: {alert['severity']}."
    })
    
    try:
        alert_dt = datetime.strptime(alert["timestamp"], "%Y-%m-%d %H:%M:%S")
        with get_db() as conn:
            all_logs = conn.execute("SELECT * FROM message_log ORDER BY timestamp ASC").fetchall()
            for log in all_logs:
                try:
                    log_dt = datetime.strptime(log["timestamp"], "%Y-%m-%d %H:%M:%S")
                    if abs((log_dt - alert_dt).total_seconds()) <= 300: # within 5 minutes
                        timeline_events.append({
                            "time": log["timestamp"],
                            "event": f"Notification to {log['sent_to']} ({log['role']})",
                            "details": f"Status: {log['status']}. Message: {log['message']}"
                        })
                except Exception:
                    pass
    except Exception as e:
        print("Timeline parsing error:", e)
        
    # Sort timeline
    timeline_events.sort(key=lambda x: x["time"])
    
    # Generate the PDF
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=15,
        alignment=1 # Center
    )
    
    h2_style = ParagraphStyle(
        'SecHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=12,
        spaceAfter=6,
        borderColor=colors.HexColor('#1e3a8a'),
        borderWidth=0.5,
        borderRadius=2,
        borderPadding=4
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        textColor=colors.HexColor('#333333'),
        leading=13
    )
    
    bold_style = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    # Header Banner block
    header_table_data = [
        [Paragraph("<b>UAV DISASTER DETECTION - SITUATION REPORT</b>", ParagraphStyle('HCol', parent=title_style, fontSize=15, textColor=colors.white, alignment=0)),
         Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ParagraphStyle('HColR', parent=body_style, textColor=colors.white, alignment=2))]
    ]
    header_table = Table(header_table_data, colWidths=[380, 160])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1e3a8a')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 15))
    
    # Summary Table
    severity_color = '#dc2626' if alert['severity'] == 'HIGH' else '#ea580c' if alert['severity'] == 'MEDIUM' else '#16a34a'
    summary_data = [
        [Paragraph("<b>Date & Time:</b>", bold_style), Paragraph(alert['timestamp'], body_style),
         Paragraph("<b>Severity Level:</b>", bold_style), Paragraph(f"<font color='{severity_color}'><b>{alert['severity']}</b></font>", body_style)],
        [Paragraph("<b>GPS Coordinates:</b>", bold_style), Paragraph(f"{alert['gps_location']['latitude']:.6f}, {alert['gps_location']['longitude']:.6f} (Alt: {alert['gps_location']['altitude']}m)", body_style),
         Paragraph("<b>Humans Detected:</b>", bold_style), Paragraph(str(alert['humans']), bold_style)],
        [Paragraph("<b>Location Name:</b>", bold_style), Paragraph(alert['location'] or "N/A", body_style),
         Paragraph("<b>Disaster Type:</b>", bold_style), Paragraph(alert.get('disaster_type') or "Natural Disaster Zone", body_style)],
    ]
    summary_table = Table(summary_data, colWidths=[100, 180, 100, 160])
    summary_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#f8fafc')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(Paragraph("Detection Summary", h2_style))
    story.append(summary_table)
    story.append(Spacer(1, 10))
    
    # Weather Details
    weather_data = [
        [Paragraph("<b>Temperature:</b>", bold_style), Paragraph(f"{temp} °C", body_style),
         Paragraph("<b>Wind Speed:</b>", bold_style), Paragraph(f"{wind} km/h", body_style),
         Paragraph("<b>Weather Condition Code:</b>", bold_style), Paragraph(str(weather_code), body_style)]
    ]
    weather_table = Table(weather_data, colWidths=[90, 90, 90, 90, 120, 70])
    weather_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0,0), (0,0), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (2,0), (2,0), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (4,0), (4,0), colors.HexColor('#f8fafc')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(Paragraph("Weather Conditions (At Location)", h2_style))
    story.append(weather_table)
    story.append(Spacer(1, 10))
    
    # Centered Snapshot Image
    if alert['image'] and os.path.exists(alert['image']):
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(alert['image'])
            w_original, h_original = pil_img.size
            max_w = 420
            max_h = 240
            ratio = min(max_w / w_original, max_h / h_original)
            new_w = int(w_original * ratio)
            new_h = int(h_original * ratio)
            
            img_obj = Image(alert['image'], width=new_w, height=new_h)
            img_obj.hAlign = 'CENTER'
            
            img_container = []
            img_container.append(Paragraph("Visual Evidence (Privacy Masked Face / Bounding Boxes)", h2_style))
            img_container.append(Spacer(1, 4))
            img_container.append(img_obj)
            img_container.append(Spacer(1, 4))
            img_container.append(Paragraph(f"<font color='#64748b'><i>Figure 1: Aerial snapshot taken at coordinates {alert['gps_location']['latitude']:.6f}, {alert['gps_location']['longitude']:.6f}</i></font>", ParagraphStyle('FigCap', parent=body_style, alignment=1, fontSize=8.5)))
            
            story.append(KeepTogether(img_container))
            story.append(Spacer(1, 10))
        except Exception as e:
            print("Error rendering image in PDF:", e)
            
    # Response Team Contacts
    team_container = []
    team_container.append(Paragraph("Assigned Response Team", h2_style))
    if response_team:
        team_table_data = [
            [Paragraph("<b>Name</b>", bold_style), Paragraph("<b>Role</b>", bold_style), Paragraph("<b>Phone</b>", bold_style), Paragraph("<b>Email</b>", bold_style)]
        ]
        for contact in response_team:
            team_table_data.append([
                Paragraph(contact["name"], body_style),
                Paragraph(contact["role"], body_style),
                Paragraph(contact["phone"], body_style),
                Paragraph(contact.get("email") or "--", body_style)
            ])
        team_table = Table(team_table_data, colWidths=[140, 110, 110, 180])
        team_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#edf2f7')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        team_container.append(team_table)
    else:
        team_container.append(Paragraph("<i>No rescue team contacts currently registered in database.</i>", body_style))
    story.append(KeepTogether(team_container))
    story.append(Spacer(1, 10))
    
    # Timeline
    timeline_container = []
    timeline_container.append(Paragraph("Alert Timeline & Notification Logs", h2_style))
    if timeline_events:
        timeline_table_data = [
            [Paragraph("<b>Time</b>", bold_style), Paragraph("<b>Event</b>", bold_style), Paragraph("<b>Description / Status</b>", bold_style)]
        ]
        for ev in timeline_events:
            timeline_table_data.append([
                Paragraph(ev["time"], body_style),
                Paragraph(ev["event"], bold_style),
                Paragraph(ev["details"], body_style)
            ])
        timeline_table = Table(timeline_table_data, colWidths=[120, 130, 290])
        timeline_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#edf2f7')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        timeline_container.append(timeline_table)
    else:
        timeline_container.append(Paragraph("<i>No timeline data available for this event.</i>", body_style))
    story.append(KeepTogether(timeline_container))
    
    # Footer drawing
    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.setStrokeColor(colors.HexColor('#e2e8f0'))
        canvas.setLineWidth(0.5)
        canvas.line(36, 30, letter[0]-36, 30)
        canvas.drawString(36, 18, "CONFIDENTIAL - UAV DISASTER RESPONDER SYSTEM")
        canvas.drawRightString(letter[0]-36, 18, f"Page {doc.page}")
        canvas.restoreState()
        
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    
    buffer.seek(0)
    return Response(
        buffer.getvalue(),
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment;filename=UAV_Situation_Report_Alert_{alert_id}.pdf"}
    )


@app.route("/status")
@login_required
def status():
    try:
        with get_db() as conn:
            alert_count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    except Exception:
        alert_count = len(state["alerts"])

    with lock:
        return jsonify({
            "human_count":      state["human_count"],
            "severity":         state["severity"],
            "gps":              state["gps"],
            "last_alert_time":  state["last_alert_time"],
            "total_detections": state["total_detections"],
            "alert_count":      alert_count,
            "camera_active":    camera_active,
            "camera_mode":      CAMERA_MODE,
        })


@app.route("/camera/toggle", methods=["POST"])
@login_required
def toggle_camera():
    global camera_active, raw_frame, output_frame
    with lock:
        camera_active = not camera_active
        if not camera_active:
            raw_frame = None
            output_frame = None
    return jsonify({"status": "success", "camera_active": camera_active})


@app.route("/camera/upload_frame", methods=["POST"])
@login_required
def upload_frame():
    global raw_frame
    if not camera_active:
        return jsonify({"error": "Camera is not active"}), 400
    
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image in request"}), 400

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            frame = cv2.resize(frame, (640, 480))
            with lock:
                raw_frame = frame
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": "Failed to decode frame"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/update_location", methods=["POST"])
@login_required
def update_location():
    global last_browser_gps
    data = request.json
    if not data or "latitude" not in data or "longitude" not in data:
        return jsonify({"error": "invalid data"}), 400
    
    lat = float(data["latitude"])
    lon = float(data["longitude"])
    alt = float(data.get("altitude") or 0)
    
    with lock:
        last_browser_gps = {
            "latitude": lat,
            "longitude": lon,
            "altitude": alt
        }
        state["gps"]["latitude"] = lat
        state["gps"]["longitude"] = lon

    def fetch_weather_async(latitude, longitude):
        try:
            w_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
            w_data = requests.get(w_url, timeout=2).json()
            if "current_weather" in w_data:
                with lock:
                    state["gps"]["weather"] = {
                        "temp": w_data["current_weather"].get("temperature", "--"),
                        "wind": w_data["current_weather"].get("windspeed", "--"),
                        "code": w_data["current_weather"].get("weathercode", "--")
                    }
        except Exception:
            pass
            
    threading.Thread(target=fetch_weather_async, args=(lat, lon), daemon=True).start()
    return jsonify({"status": "success"})


@app.route("/alerts")
@login_required
def alerts():
    try:
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM alerts ORDER BY timestamp DESC, id DESC").fetchall()
        alert_list = [db_row_to_alert_dict(row) for row in rows]
        return jsonify(alert_list)
    except Exception as e:
        print("Database error in /alerts:", e)
        with lock:
            return jsonify(state["alerts"])


@app.route("/alert", methods=["POST"])
def receive_alert():
    data = request.json
    if data:
        data.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        gps = data.get("gps_location") or data.get("gps")
        if gps and "latitude" in gps and "longitude" in gps:
            lat = gps["latitude"]
            lon = gps["longitude"]
            if "location" not in data or not data["location"]:
                data["location"] = gps.get("location") or reverse_geocode(lat, lon)
        add_or_merge_alert(data)
    return jsonify({"status": "received"})


@app.route("/delete_alert", methods=["POST"])
@login_required
def delete_alert():
    data = request.json or {}
    alert_id = data.get("id")
    timestamp = data.get("timestamp")
    try:
        with get_db() as conn:
            if alert_id is not None and str(alert_id) != 'null':
                conn.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
            elif timestamp:
                conn.execute("DELETE FROM alerts WHERE timestamp=?", (timestamp,))
            conn.commit()
            

        with lock:
            if alert_id is not None and str(alert_id) != 'null':
                state["alerts"] = [a for a in state["alerts"] if a.get("id") != alert_id]
            elif timestamp:
                state["alerts"] = [a for a in state["alerts"] if a.get("timestamp") != timestamp]
                
        return jsonify({"status": "success", "message": "Alert deleted successfully"})
    except Exception as e:
        print("Error deleting alert:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/alerts/clear", methods=["DELETE"])
@login_required
def clear_alerts():
    try:
        with get_db() as conn:
            conn.execute("DELETE FROM alerts")
            conn.commit()
        with lock:
            state["alerts"] = []
        return jsonify({"status": "success", "message": "All alert logs cleared successfully"})
    except Exception as e:
        print("Error clearing alert logs:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/contacts", methods=["GET"])
@login_required
def get_contacts():
    role = request.args.get("role", "").strip()
    with get_db() as conn:
        if role:
            rows = conn.execute(
                "SELECT * FROM contacts WHERE LOWER(role)=? ORDER BY role, name",
                (role.lower(),)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM contacts ORDER BY role, name"
            ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/contacts", methods=["POST"])
@login_required
def add_contact():
    data = request.json
    if not data or not data.get("name") or not data.get("phone"):
        return jsonify({"error": "name and phone required"}), 400
    contact = {
        "id":         datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "name":       data["name"].strip(),
        "phone":      data["phone"].strip(),
        "email":      data.get("email", "").strip(),
        "role":       data.get("role", "Rescue Team").strip(),
        "address":    data.get("address", "").strip(),
        "notes":      data.get("notes", "").strip(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with get_db() as conn:
        conn.execute(
            "INSERT INTO contacts (id,name,phone,email,role,address,notes,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (contact["id"], contact["name"], contact["phone"], contact["email"],
             contact["role"], contact["address"], contact["notes"], contact["created_at"])
        )
        conn.commit()
    return jsonify({"status": "added", "contact": contact})


@app.route("/contacts/<cid>", methods=["DELETE"])
@login_required
def delete_contact(cid):
    with get_db() as conn:
        conn.execute("DELETE FROM contacts WHERE id=?", (cid,))
        conn.commit()
    return jsonify({"status": "deleted"})


@app.route("/contacts/<cid>", methods=["PUT"])
@login_required
def update_contact(cid):
    data = request.json
    with get_db() as conn:
        conn.execute(
            "UPDATE contacts SET name=?, phone=?, email=?, role=?, address=?, notes=? WHERE id=?",
            (data.get("name"), data.get("phone"), data.get("email", ""), data.get("role"),
             data.get("address", ""), data.get("notes", ""), cid)
        )
        conn.commit()
    return jsonify({"status": "updated"})


@app.route("/contacts/stats", methods=["GET"])
@login_required
def contact_stats():
    with get_db() as conn:
        total      = conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
        victims    = conn.execute("SELECT COUNT(*) FROM contacts WHERE LOWER(role)='victim'").fetchone()[0]
        volunteers = conn.execute("SELECT COUNT(*) FROM contacts WHERE LOWER(role)='volunteer'").fetchone()[0]
        rescue     = conn.execute("SELECT COUNT(*) FROM contacts WHERE LOWER(role)='rescue team'").fetchone()[0]
    return jsonify({"total": total, "victims": victims, "volunteers": volunteers, "rescue_team": rescue})


@app.route("/send_message", methods=["POST"])
@login_required
def send_message():
    data    = request.json
    message = data.get("message", "").strip()
    target  = data.get("target", "all")
    if not message:
        return jsonify({"error": "message required"}), 400

    with get_db() as conn:
        if target == "all":
            rows = conn.execute("SELECT * FROM contacts ORDER BY role, name").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM contacts WHERE LOWER(role)=? ORDER BY name",
                (target.lower(),)
            ).fetchall()
        contacts = [dict(r) for r in rows]

    if not contacts:
        return jsonify({"status": "ok", "sent_to": [], "failed": [], "total": 0})

    sent_names  = []
    failed_names = []

    def send_one(c):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = "[UAV ALERT] Role:{} | {} | {}".format(c["role"], message, ts)
        ok, status_msg = send_sms(c["phone"], body)
        final_status = "Sent" if ok else "Failed: " + status_msg
        with get_db() as db:
            db.execute(
                "INSERT INTO message_log (sent_to,phone,role,message,status,timestamp) VALUES (?,?,?,?,?,?)",
                (c["name"], c["phone"], c["role"], message, final_status, ts)
            )
            db.commit()
        with lock:
            state["messages"].insert(0, {
                "timestamp": ts, "to": c["name"], "phone": c["phone"],
                "role": c["role"], "message": message, "status": final_status,
            })
        if ok:
            sent_names.append(c["name"])
        else:
            failed_names.append(c["name"])

    with ThreadPoolExecutor(max_workers=min(10, len(contacts))) as executor:
        executor.map(send_one, contacts)
    with lock:
        state["messages"] = state["messages"][:100]

    return jsonify({"status": "ok", "sent_to": sent_names, "failed": failed_names, "total": len(contacts)})


@app.route("/send_shortage_alert", methods=["POST"])
@login_required
def send_shortage_alert():
    data     = request.json
    res_name = data.get("name", "Unknown")
    res_cat  = data.get("cat", "")
    qty      = data.get("qty", 0)
    min_qty  = data.get("min", 0)
    loc      = data.get("loc", "")

    message_en = (
        "[SHORTAGE ALERT] {} {} is critically low! "
        "Available: {} | Minimum Required: {} | Location: {}. "
        "Immediate resupply needed."
    ).format(res_cat, res_name, qty, min_qty, loc or "N/A")

    with get_db() as conn:
        contacts = [dict(r) for r in conn.execute("SELECT * FROM contacts").fetchall()]

    if not contacts:
        return jsonify({"status": "no_contacts", "sent": 0})

    sent, failed = [], []

    def send_one(c):
        ok, _ = send_sms(c["phone"], message_en)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_db() as db:
            db.execute(
                "INSERT INTO message_log (sent_to,phone,role,message,status,timestamp) VALUES (?,?,?,?,?,?)",
                (c["name"], c["phone"], c["role"], message_en, "Sent" if ok else "Failed", ts)
            )
            db.commit()
        (sent if ok else failed).append(c["name"])

    with ThreadPoolExecutor(max_workers=min(10, len(contacts))) as executor:
        executor.map(send_one, contacts)

    return jsonify({"status": "ok", "sent": len(sent), "failed": len(failed)})



@app.route("/messages/clear", methods=["DELETE"])
@login_required
def clear_messages():
    with get_db() as conn:
        conn.execute("DELETE FROM message_log")
        conn.commit()
    with lock:
        state["messages"] = []
    return jsonify({"status": "cleared"})


@app.route("/messages", methods=["GET"])
@login_required
def get_messages():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM message_log ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


def start_ssh_tunnel():
    import subprocess
    import re
    import threading
    import time

    def tunnel_thread():
        url_pattern = re.compile(r"https://[a-zA-Z0-9-.]+\.lhr\.life")
        while True:
            try:
                print("[Tunnel] Starting SSH tunnel to localhost.run...", flush=True)
                cmd = ["ssh", "-tt", "-o", "StrictHostKeyChecking=no", "-R", "80:127.0.0.1:5000", "nokey@localhost.run"]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                
                for line in process.stdout:
                    print(f"[Tunnel] {line.strip()}", flush=True)
                    match = url_pattern.search(line)
                    if match:
                        public_url = match.group(0)
                        print(f"[Tunnel] Detected active Public URL: {public_url}", flush=True)
                        os.environ["PUBLIC_SERVER_URL"] = public_url
                
                print("[Tunnel] SSH tunnel disconnected. Retrying in 5 seconds...", flush=True)
            except Exception as e:
                print(f"[Tunnel] Error in SSH tunnel process: {e}", flush=True)
            time.sleep(5)

    threading.Thread(target=tunnel_thread, daemon=True).start()


if __name__ == "__main__":
    if CAMERA_MODE == "local":
        start_ssh_tunnel()
    try:
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM alerts ORDER BY timestamp DESC, id DESC LIMIT 50").fetchall()
        state["alerts"] = [db_row_to_alert_dict(row) for row in rows]
        print("Loaded {} alerts from database on startup".format(len(state["alerts"])))
    except Exception as e:
        print("Failed to load alerts on startup:", e)

    threading.Thread(target=capture_loop,   daemon=True).start()
    threading.Thread(target=detection_loop, daemon=True).start()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)