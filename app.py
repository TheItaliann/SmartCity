from flask import Flask, Response
from picamera2 import Picamera2
import threading
import queue
import cv2
import signal
import numpy as np
from simple_pid import PID

# Flask-App erstellen
app = Flask(__name__)

# Kamera initialisieren
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (640,480)})
picam2.configure(video_config)
picam2.start()

# Frame-Queue
frame_queue = queue.Queue(maxsize=1)

# Signal-Handler zum Freigeben der Kamera
def cleanup_camera(signum, frame):
    picam2.stop()
    print("Kamera freigegeben.")
    exit(0)

signal.signal(signal.SIGINT, cleanup_camera)
signal.signal(signal.SIGTERM, cleanup_camera)

# Frames kontinuierlich erfassen
def capture_frames():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        low_b = np.array([0, 0, 0], dtype=np.uint8)
        high_b = np.array([5, 5, 5], dtype=np.uint8)
        mask = cv2.inRange(frame, low_b, high_b)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
        
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put(frame)

# Hintergrund-Thread für Frame-Erfassung
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True
capture_thread.start()

# MJPEG-Stream erzeugen
def grab_frames():
    while True:
        frame = frame_queue.get()
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/stream.mjpg')
def video_feed():
    """Route für den MJPEG-Stream."""
    return Response(grab_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """HTML-Indexseite."""
    return '''
    <html>
    <head>
        <title>Raspberry Pi Camera Stream</title>
    </head>
    <body>
        <h1>Raspberry Pi Camera Streaming</h1>
        <img src="/stream.mjpg" width="640" height="480" />
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
