# Import necessary libraries and modules
from flask import Flask, Response           # Flask: For running a web server and handling HTTP responses
from picamera2 import Picamera2              # Picamera2: To control the camera
import threading                             # threading: To run processes in the background
import queue                                 # queue: To safely pass data between threads
import cv2                                   # OpenCV: For image processing tasks
import signal                                # signal: To handle OS signals (e.g., for clean shutdown)
import numpy as np                           # NumPy: For mathematical operations and handling arrays
import RPi.GPIO as GPIO                      # RPi.GPIO: To control GPIO pins on the Raspberry Pi
from simple_pid import PID                   # PID: To implement a PID controller for motor control
from time import sleep                       # sleep: To add delays
import socket                                # socket: For network communication
import json                                  # json: For processing JSON formatted data

# --- Definition of the Motor Control Class ---
class Driver:
    """
    This class controls the vehicle's motors.
    It sets up the necessary GPIO pins, starts PWM signals,
    and provides methods to move and stop the vehicle.
    """
    def __init__(self, pin1: int, pin2: int, pin3: int, pin4: int) -> None:
        # Disable warnings and set the pin numbering mode (BOARD mode)
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        # Set up the four pins as outputs
        GPIO.setup(pin1, GPIO.OUT)
        GPIO.setup(pin2, GPIO.OUT)
        GPIO.setup(pin3, GPIO.OUT)
        GPIO.setup(pin4, GPIO.OUT)
        
        # Create PWM objects for forward and backward movement on both sides
        self.leftForward = GPIO.PWM(pin2, 1000)
        self.leftBackward = GPIO.PWM(pin1, 1000)
        self.rightForward = GPIO.PWM(pin4, 1000)
        self.rightBackward = GPIO.PWM(pin3, 1000)
        
        # Start PWM signals with 0% duty cycle (motors are initially stopped)
        self.leftForward.start(0)
        self.leftBackward.start(0)
        self.rightForward.start(0)
        self.rightBackward.start(0)
    
        # Initialize current speed and direction states
        self.current_speed = 0
        self.current_direction = 0
        self.is_reversing = False

    def move(self, left_speed, right_speed):
        """
        Moves the vehicle by adjusting the speed of the left and right motors.
        
        Parameters:
        - left_speed: Speed for the left motor (positive = forward, negative = backward)
        - right_speed: Speed for the right motor (positive = forward, negative = backward)
        """
        # Control for the left motor
        if left_speed >= 0:
            self.leftForward.ChangeDutyCycle(left_speed)  # Move forward
            self.leftBackward.ChangeDutyCycle(0)
        else:
            self.leftForward.ChangeDutyCycle(0)
            self.leftBackward.ChangeDutyCycle(abs(left_speed))  # Move backward

        # Control for the right motor
        if right_speed >= 0:
            self.rightForward.ChangeDutyCycle(right_speed)  # Move forward
            self.rightBackward.ChangeDutyCycle(0)
        else:
            self.rightForward.ChangeDutyCycle(0)
            self.rightBackward.ChangeDutyCycle(abs(right_speed))  # Move backward

    def stop(self):
        """
        Stops the vehicle by setting all motor speeds to zero.
        """
        self.leftForward.ChangeDutyCycle(0)
        self.leftBackward.ChangeDutyCycle(0)
        self.rightForward.ChangeDutyCycle(0)
        self.rightBackward.ChangeDutyCycle(0)


# --- Initialize the Motor Control ---
# Define the pins (according to BOARD numbering) for the motor driver
driver = Driver(pin1=35, pin2=33, pin3=12, pin4=32)  

# --- Socket Server Configuration ---
SERVER_HOST = "192.168.0.177"  # The IP address for the server
SERVER_PORT = 8400             # The port on which commands are received

# --- Initialize the Camera ---
picam2 = Picamera2()  # Create a camera object
# Configure the camera for video capture with a resolution of 640x480 pixels
video_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(video_config)
picam2.start()  # Start the camera

# --- PID Controller Setup for Motor Control ---
setpoint_x = 310  # Desired setpoint for the steering center (position in the image)
# Initialize the PID controller with specific parameters:
# Kp, Ki, Kd, setpoint, and output limits (-15 to 15)
pid = PID(0.4, 0.19, 0.017, setpoint=setpoint_x, output_limits=(-15, 15))

# --- Signal Handler for a Clean Shutdown ---
def cleanup(signum, frame):
    """
    This function is called when a termination signal is received.
    It stops the camera, stops the motors, and cleans up the GPIO pins.
    """
    picam2.stop()    # Stop the camera
    driver.stop()    # Stop the motors
    GPIO.cleanup()   # Release the GPIO resources
    print("Camera and GPIOs have been released.")
    exit(0)

# Catch SIGINT (Ctrl+C) and SIGTERM signals to perform cleanup
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# --- Create a Flask Web Application ---
app = Flask(__name__)

# --- Create a Queue to Store Image Frames ---
frame_queue = queue.Queue(maxsize=1)

# Global flag to indicate whether the driving algorithm is enabled (enabled by default)
driving_enabled = True

# --- Function to Capture Frames, Detect Lines, and Compute Control ---
def capture_frames():
    """
    This function runs in an infinite loop:
    - Captures images from the camera.
    - Processes the image to detect the region of interest (the road).
    - Determines a steering point based on detected lines.
    - Calculates the necessary correction using a PID controller.
    - Adjusts motor speeds and sends commands to the driver.
    - Puts the processed image into a queue for later streaming.
    """
    while True:
        # Capture an image from the camera and convert it from RGB to BGR (OpenCV standard)
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]

        # --- Define the Region Of Interest (ROI) for the road ---
        roi_mask = np.zeros((height, width), dtype=np.uint8)  # Create an empty mask
        roi_corners = np.array([[ 
            (0, height),                         # Bottom-left point
            (width, height),                     # Bottom-right point
            (int(width * 0.85), int(height * 0.5)),# Top-right point
            (int(width * 0.08), int(height * 0.65)) # Top-left point
        ]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_corners, 255)  # Fill the ROI area with white
        cv2.polylines(frame, roi_corners, isClosed=True, color=(255, 0, 0), thickness=2)  # Draw the ROI outline on the image

        # --- Process the image within the ROI ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # Convert the image to grayscale
        gray_roi = cv2.bitwise_and(gray, roi_mask)              # Apply the ROI mask
        _, mask_binary = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)  # Apply thresholding to detect white areas
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours (lines)
        # Sort the detected contours by area and keep at most the two largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # Set the "lookahead" point (the point used for steering) at 98% of the image height
        steer_y = int(height * 0.98)

        # --- Determine the steering point based on the detected contours ---
        if len(contours) > 0:
            xs = []
            for cnt in contours:
                if len(cnt) >= 2:
                    # Fit a line to the contour to determine its direction
                    vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx = float(vx)
                    vy = float(vy)
                    x0 = float(x0)
                    y0 = float(y0)
                    # Prevent division by zero; if vy is too small, set x_at_bottom directly
                    if abs(vy) < 1e-6:
                        x_at_bottom = int(x0)
                    else:
                        # Calculate the x-coordinate of the line at the lookahead point
                        x_at_bottom = int(((steer_y - y0) * (vx / vy)) + x0)
                    xs.append(x_at_bottom)
            # If two contours are found, compute the average as the steering point
            if len(xs) == 2:
                cx = int(np.mean(xs))
            # If only one contour is detected, apply an offset for sharper correction
            elif len(xs) == 1:
                detected_line_x = xs[0]
                offset = 300  # Offset value for correction
                if detected_line_x > width / 2:
                    cx = detected_line_x - offset
                else:
                    cx = detected_line_x + offset
            else:
                cx = setpoint_x  # Default value if no line is detected
            # Draw a small circle at the calculated steering point for visualization
            cv2.circle(frame, (cx, steer_y), 5, (255, 255, 255), -1)
        else:
            cx = setpoint_x  # Default value if no contours are found

        # --- Calculate the necessary correction using the PID controller ---
        correction = pid(cx)

        # **Dynamic Speed Adjustment:**
        # Reduce the base speed when larger corrections are needed to allow more reaction time in sharp curves.
        if abs(correction) > 5:  # Threshold value – can be adjusted
            base_speed = 25
        elif abs(correction) > 10:
            base_speed = 20
        elif abs(correction) > 15:
            base_speed = 10
        else:
            base_speed = 30

        # Compute individual speeds for the left and right motors
        left_motor_speed = base_speed + correction
        right_motor_speed = base_speed - correction

        # Limit the motor speeds to a defined range (between 0 and 40)
        left_motor_speed = max(0, min(40, left_motor_speed))
        right_motor_speed = max(0, min(40, right_motor_speed))
        
        if driving_enabled:
            # If the driving algorithm is active, move the vehicle with the calculated speeds
            # (Debug output can be enabled here to check the values)
            #print(f"Driving enabled: left_motor_speed={left_motor_speed}, right_motor_speed={right_motor_speed}")
            driver.move(left_motor_speed, right_motor_speed)
        else:
            # If the driving algorithm is deactivated, stop the vehicle
            driver.stop()

        # Draw the detected contours and the steering point (for visualization) on the image
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        cv2.circle(frame, (cx, steer_y), 5, (0, 0, 255), -1)

        # Ensure that the queue always holds only the latest frame
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put(frame)

# --- Start a Background Thread for Frame Capture ---
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True  # Daemon thread will exit when the main program ends
capture_thread.start()

# --- Function to Handle Commands Received via Socket ---
def handle_client(conn):
    """
    Processes incoming commands (either in JSON format or plain text) over a socket connection.
    - "STOP": Deactivates the driving algorithm and stops the vehicle.
    - "START": Reactivates the driving algorithm.
    
    Parameter:
    - conn: The socket connection to the client.
    """
    global driving_enabled
    with conn:
        while True:
            data = conn.recv(1024)  # Read up to 1024 bytes from the connection
            if not data:
                break  # Exit loop if no data is received
            raw_message = data.decode().strip()  # Decode the received bytes into a string
            print(f"Raw message received: {raw_message}")
            try:
                payload = json.loads(raw_message)  # Try to parse the string as JSON
                command = payload.get("command")
            except json.JSONDecodeError:
                command = raw_message  # If not valid JSON, treat the string as a plain command
            print(f"Processed command: {command}")
            if command == "STOP":
                driving_enabled = False  # Deactivate the driving algorithm
                driver.stop()            # Stop the vehicle
                print("Driving algorithm deactivated.")
            elif command == "START":
                driving_enabled = True   # Activate the driving algorithm
                print("Driving algorithm resumed.")
            else:
                print(f"Unknown command received: {command}")

# --- Set Up the Socket Server ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_HOST, SERVER_PORT))  # Bind the socket to the specified IP address and port
server_socket.listen(1)  # Listen for incoming connections (max. 1 simultaneous connection)
print(f"Socket server listening on {SERVER_HOST}:{SERVER_PORT}")

# --- Function That Waits for New Socket Connections ---
def socket_server():
    """
    Continuously waits for incoming socket connections.
    When a new connection is accepted, a new thread is started to handle the client.
    """
    while True:
        conn, addr = server_socket.accept()  # Accept a new connection
        print(f"Accepted connection from {addr}")
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

# Start a thread for the socket server
threading.Thread(target=socket_server, daemon=True).start()

# --- MJPEG Streaming: Creates a Continuous Stream of the Latest Frames ---
def grab_frames():
    """
    A generator function that continuously retrieves image frames from the queue,
    encodes them as JPEG, and yields them as part of an MJPEG stream.
    """
    while True:
        frame = frame_queue.get()           # Get the latest frame from the queue
        _, jpeg = cv2.imencode('.jpg', frame) # Encode the frame in JPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# --- Flask Routes for the Web Stream ---
@app.route('/stream.mjpg')
def video_feed():
    """
    Flask route that returns the MJPEG stream as an HTTP response.
    Clients can use this route to view the live stream in their browsers.
    """
    return Response(grab_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """
    Flask route that returns a simple HTML page with the embedded MJPEG stream.
    This page serves as the user interface for viewing the live stream.
    """
    return '''
    <html>
    <head>
        <title>Line Detection Stream</title>
    </head>
    <body>
        <h1>Line Detection (Road Only)</h1>
        <img src="/stream.mjpg" width="640" height="480" />
    </body>
    </html>
    '''

# --- Start the Flask Web Server When This Script is Run Directly ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
