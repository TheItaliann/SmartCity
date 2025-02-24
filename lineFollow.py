from flask import Flask, Response
from picamera2 import Picamera2
import threading
import queue
import cv2
import signal
import numpy as np
import RPi.GPIO as GPIO
from simple_pid import PID
from time import sleep
import socket
import json

# Motorsteuerung mit deiner Driver-Klasse
class Driver:
    def __init__(self, pin1: int, pin2: int, pin3: int, pin4: int) -> None:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin1, GPIO.OUT)
        GPIO.setup(pin2, GPIO.OUT)
        GPIO.setup(pin3, GPIO.OUT)
        GPIO.setup(pin4, GPIO.OUT)
        
        self.leftForward = GPIO.PWM(pin2, 1000)
        self.leftBackward = GPIO.PWM(pin1, 1000)
        self.rightForward = GPIO.PWM(pin4, 1000)
        self.rightBackward = GPIO.PWM(pin3, 1000)
        
        self.leftForward.start(0)
        self.leftBackward.start(0)
        self.rightForward.start(0)
        self.rightBackward.start(0)
    
        self.current_speed = 0
        self.current_direction = 0
        self.is_reversing = False

    def move(self, left_speed, right_speed):
        """Bewegt das Auto basierend auf den berechneten Geschwindigkeiten"""
        if left_speed >= 0:
            self.leftForward.ChangeDutyCycle(left_speed)
            self.leftBackward.ChangeDutyCycle(0)
        else:
            self.leftForward.ChangeDutyCycle(0)
            self.leftBackward.ChangeDutyCycle(abs(left_speed))

        if right_speed >= 0:
            self.rightForward.ChangeDutyCycle(right_speed)
            self.rightBackward.ChangeDutyCycle(0)
        else:
            self.rightForward.ChangeDutyCycle(0)
            self.rightBackward.ChangeDutyCycle(abs(right_speed))

    def stop(self):
        """Stoppt das Auto"""
        self.leftForward.ChangeDutyCycle(0)
        self.leftBackward.ChangeDutyCycle(0)
        self.rightForward.ChangeDutyCycle(0)
        self.rightBackward.ChangeDutyCycle(0)


# Motor-Pins definieren (Board-Nummern!)
driver = Driver(pin1=35, pin2=33, pin3=12, pin4=32)  

SERVER_HOST = "192.168.0.177"
SERVER_PORT = 8400

# Kamera initialisieren
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(video_config)
picam2.start()

# PID-Setup für Motorsteuerung
setpoint_x = 310  # Standard-Sollwert für die Lenkmitte
pid = PID(0.4, 0.19, 0.017, setpoint=setpoint_x, output_limits=(-15, 15))

# Signal-Handler zum Beenden
def cleanup(signum, frame):
    picam2.stop()
    driver.stop()
    GPIO.cleanup()
    print("Kamera und GPIOs freigegeben.")
    exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# Flask-App erstellen
app = Flask(__name__)

# Frame-Queue für den Stream
frame_queue = queue.Queue(maxsize=1)

# Global flag to control driving via algorithm – set to True by default
driving_enabled = True

# Frames erfassen, Linien erkennen und Steuerung berechnen
def capture_frames():
    while True:
        # Bild von der Kamera holen und in BGR umwandeln
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]

        # --- Definition des ROI (Fahrbahn) ---
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_corners = np.array([[ 
            (0, height),                         # unterer linker Punkt
            (width, height),                     # unterer rechter Punkt
            (int(width * 0.85), int(height * 0.5)),# oberer rechter Punkt
            (int(width * 0.08), int(height * 0.65)) # oberer linker Punkt
        ]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_corners, 255)
        cv2.polylines(frame, roi_corners, isClosed=True, color=(255, 0, 0), thickness=2)

        # --- Verarbeitung nur im ROI ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.bitwise_and(gray, roi_mask)
        _, mask_binary = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sortiere nach Fläche und nutze maximal die 2 größten Konturen
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # Setze den Lookahead-Punkt bei 98% der Bildhöhe (wie bisher)
        steer_y = int(height * 0.98)

        # Ermittlung des Lenkpunktes:
        if len(contours) > 0:
            xs = []
            for cnt in contours:
                if len(cnt) >= 2:
                    # Linienanpassung, um den Linienverlauf zu ermitteln
                    vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx = float(vx)
                    vy = float(vy)
                    x0 = float(x0)
                    y0 = float(y0)
                    if abs(vy) < 1e-6:
                        x_at_bottom = int(x0)
                    else:
                        x_at_bottom = int(((steer_y - y0) * (vx / vy)) + x0)
                    xs.append(x_at_bottom)
            if len(xs) == 2:
                cx = int(np.mean(xs))
            elif len(xs) == 1:
                detected_line_x = xs[0]
                offset = 300  # Erhöhter Versatz für schärfere Korrektur
                if detected_line_x > width / 2:
                    cx = detected_line_x - offset
                else:
                    cx = detected_line_x + offset
            else:
                cx = setpoint_x
            cv2.circle(frame, (cx, steer_y), 5, (255, 255, 255), -1)
        else:
            cx = setpoint_x

        # PID-Korrektur berechnen
        correction = pid(cx)


        # **Dynamische Geschwindigkeitsanpassung:**
        # Bei hohen Korrekturen (starke Abweichung) wird die Basisgeschwindigkeit reduziert,
        # um in engen Kurven mehr Reaktionszeit zu ermöglichen.
        if abs(correction) > 5:  # Schwellenwert – ggf. anpassen
            base_speed = 25
        elif abs(correction) > 10:
            base_speed = 20
        elif abs(correction) > 15:
            base_speed = 10
        else:
            base_speed = 30

        left_motor_speed = base_speed + correction
        right_motor_speed = base_speed - correction

        left_motor_speed = max(0, min(40, left_motor_speed))
        right_motor_speed = max(0, min(40, right_motor_speed))
        
        if driving_enabled:
            # Debug output to confirm speeds before driving
            #print(f"Driving enabled: left_motor_speed={left_motor_speed}, right_motor_speed={right_motor_speed}")
            driver.move(left_motor_speed, right_motor_speed)
        else:
            driver.stop()

        # Zeichne die gefundenen Konturen und den Lenkpunkt (rot) zur Visualisierung
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        cv2.circle(frame, (cx, steer_y), 5, (0, 0, 255), -1)

        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put(frame)

# Hintergrund-Thread für die Frame-Erfassung
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True
capture_thread.start()

def handle_client(conn):
    global driving_enabled
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break 
            raw_message = data.decode().strip()
            print(f"Raw message received: {raw_message}")
            try:
                payload = json.loads(raw_message)
                command = payload.get("command")
            except json.JSONDecodeError:
                command = raw_message
            print(f"Processed command: {command}")
            if command == "STOP":
                driving_enabled = False
                driver.stop()
                print("Driving algorithm deactivated.")
            elif command == "START":
                driving_enabled = True
                print("Driving algorithm resumed.")
            else:
                print(f"Unknown command received: {command}")

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(1)
print(f"Socket server listening on {SERVER_HOST}:{SERVER_PORT}")


def socket_server():
    while True:
        conn, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()


threading.Thread(target=socket_server, daemon=True).start()

# MJPEG-Stream erzeugen
def grab_frames():
    while True:
        frame = frame_queue.get()
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/stream.mjpg')
def video_feed():
    return Response(grab_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Linienerkennung Stream</title>
    </head>
    <body>
        <h1>Linienerkennung (nur Fahrbahn)</h1>
        <img src="/stream.mjpg" width="640" height="480" />
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
