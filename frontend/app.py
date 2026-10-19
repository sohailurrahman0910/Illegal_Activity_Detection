import os
import threading
import time
from datetime import datetime
from typing import List

import cv2
import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from playsound import playsound
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from ultralytics import YOLO

from database import Base, SessionLocal, engine
from model import prediction

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "model", "best5.pt")
ALARM_PATH = os.path.join(PROJECT_DIR, "alram", "alram.mp3")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# ---------- Settings (change these if needed) ----------
ALERT_CLASSES = ["Violence", "Smoking", "Weapon"]
IMAGE_SIZE = 320             # smaller = faster (try 416 for better accuracy)
MIN_CONFIDENCE = 0.30        # ignore weak detections
SAVE_EVERY_SECONDS = 1       # save to database once per second
ALARM_COOLDOWN_SECONDS = 5   # minimum gap between alarms

# ---------- App setup ----------
app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATES_DIR)
Base.metadata.create_all(bind=engine)
model = YOLO(MODEL_PATH)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- Alarm and database helpers ----------
last_alarm_time = 0.0


def play_alarm():
    try:
        playsound(ALARM_PATH)
    except Exception as error:
        print("Alarm error:", error)


def trigger_alarm(label):
    global last_alarm_time
    now = time.time()
    if now - last_alarm_time >= ALARM_COOLDOWN_SECONDS:
        last_alarm_time = now
        print(f"Alert! Detected: {label}")
        threading.Thread(target=play_alarm, daemon=True).start()


def save_detections(detections):
    """Save detections. Errors are printed but never stop the camera."""
    db = SessionLocal()
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for label, confidence in detections:
            db.add(prediction(name=label, accuracy=f"{confidence:.2f}", time=timestamp))
        db.commit()
    except Exception as error:
        db.rollback()
        print("Database error:", error)
    finally:
        db.close()


# ---------- Camera worker (one thread does all the work) ----------
class CameraWorker:
    def __init__(self):
        self.cap = None
        self.thread = None
        self.running = False
        self.jpeg = None          # latest frame, ready to send to the browser
        self.last_saved = 0.0

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.jpeg = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.jpeg = None

    def _loop(self):
        failed_reads = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads > 100:
                    print("Camera not giving frames. Stopping.")
                    self.running = False
                time.sleep(0.05)
                continue
            failed_reads = 0

            try:
                results = model(frame, imgsz=IMAGE_SIZE, conf=MIN_CONFIDENCE, verbose=False)
                annotated = results[0].plot()

                detections = []
                for box in results[0].boxes:
                    label = model.names[int(box.cls)]
                    detections.append((label, float(box.conf)))
                    if label in ALERT_CLASSES:
                        trigger_alarm(label)

                now = time.time()
                if detections and now - self.last_saved >= SAVE_EVERY_SECONDS:
                    save_detections(detections)
                    self.last_saved = now

                ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.jpeg = buffer.tobytes()
            except Exception as error:
                print("Detection error:", error)   # keep the camera running


camera = CameraWorker()


def stream_frames():
    """Send the latest frame to the browser (does no heavy work itself)."""
    last_sent = None
    while camera.running:
        frame = camera.jpeg
        if frame is not None and frame is not last_sent:
            last_sent = frame
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        else:
            time.sleep(0.01)


# ---------- Routes ----------
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/start_video")
def start_video():
    camera.start()
    return {"message": "Video started"}


@app.get("/stop_video")
def stop_video():
    camera.stop()
    return {"message": "Video stopped"}


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        stream_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    accuracy: str
    time: str


@app.get("/predictions", response_model=List[PredictionResponse])
def get_predictions(db: Session = Depends(get_db)):
    return db.query(prediction).order_by(prediction.id.desc()).limit(50).all()


@app.get("/class-counts")
def get_class_counts(db: Session = Depends(get_db)):
    counts = (
        db.query(prediction.name, func.count(prediction.name).label("count"))
        .group_by(prediction.name)
        .all()
    )
    return [{"name": name, "count": count} for name, count in counts]


# ---------- Start the server ----------
if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)