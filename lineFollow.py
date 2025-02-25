# =============================================================================
# IMPORTS
# =============================================================================
# Import libraries that provide various functionalities required by this program.
from flask import Flask, Response           # Flask: For running a web server and sending HTTP responses.
from picamera2 import Picamera2              # Picamera2: To control the Raspberry Pi camera.
import threading                             # threading: To run tasks concurrently (in the background).
import queue                                 # queue: To safely transfer data (frames) between threads.
import cv2                                   # OpenCV: For processing images (converting colors, drawing shapes, etc.).
import signal                                # signal: To handle operating system signals (for clean shutdown).
import numpy as np                           # NumPy: For numerical operations and handling arrays.
import RPi.GPIO as GPIO                      # RPi.GPIO: To control the Raspberry Pi's GPIO pins (for motor control).
from simple_pid import PID                   # PID: For implementing a PID controller (used for motor control).
from time import sleep                       # sleep: To add delays in execution.
import socket                                # socket: For network communication (e.g., receiving commands).
import json                                  # json: For processing data stored in JSON format.

# =============================================================================
# MOTOR CONTROL CLASS DEFINITION
# =============================================================================
class Driver:
    """
    This class controls the vehicle's motors.
    It sets up the necessary GPIO pins, starts PWM signals,
    and provides methods to move the vehicle or stop it.
    """
    def __init__(self, pin1: int, pin2: int, pin3: int, pin4: int) -> None:
        # Disable GPIO warnings and set the pin numbering mode (BOARD mode uses physical pin numbers).
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        # Set up the four specified pins as outputs.
        GPIO.setup(pin1, GPIO.OUT)
        GPIO.setup(pin2, GPIO.OUT)
        GPIO.setup(pin3, GPIO.OUT)
        GPIO.setup(pin4, GPIO.OUT)
        
        # Create PWM (Pulse Width Modulation) objects for controlling motor speed and direction.
        self.leftForward = GPIO.PWM(pin2, 1000)
        self.leftBackward = GPIO.PWM(pin1, 1000)
        self.rightForward = GPIO.PWM(pin4, 1000)
        self.rightBackward = GPIO.PWM(pin3, 1000)
        
        # Start PWM signals with a duty cycle of 0% (motors are stopped initially).
        self.leftForward.start(0)
        self.leftBackward.start(0)
        self.rightForward.start(0)
        self.rightBackward.start(0)
    
        # Initialize motor state variables.
        self.current_speed = 0
        self.current_direction = 0
        self.is_reversing = False

    def move(self, left_speed, right_speed):
        """
        Moves the vehicle by adjusting the speed of the left and right motors.
        
        Parameters:
        - left_speed: Speed for the left motor (positive value for forward, negative for backward).
        - right_speed: Speed for the right motor (positive value for forward, negative for backward).
        """
        # Control the left motor:
        if left_speed >= 0:
            self.leftForward.ChangeDutyCycle(left_speed)  # Set speed for forward motion.
            self.leftBackward.ChangeDutyCycle(0)
        else:
            self.leftForward.ChangeDutyCycle(0)
            # Use the absolute value for backward motion.
            self.leftBackward.ChangeDutyCycle(abs(left_speed))

        # Control the right motor:
        if right_speed >= 0:
            self.rightForward.ChangeDutyCycle(right_speed)  # Set speed for forward motion.
            self.rightBackward.ChangeDutyCycle(0)
        else:
            self.rightForward.ChangeDutyCycle(0)
            # Use the absolute value for backward motion.
            self.rightBackward.ChangeDutyCycle(abs(right_speed))

    def stop(self):
        """
        Stops the vehicle by setting the speed of all motors to zero.
        """
        self.leftForward.ChangeDutyCycle(0)
        self.leftBackward.ChangeDutyCycle(0)
        self.rightForward.ChangeDutyCycle(0)
        self.rightBackward.ChangeDutyCycle(0)

# =============================================================================
# INITIALIZE MOTOR CONTROL
# =============================================================================
# Create a Driver object with the specified GPIO pins (using BOARD numbering).
driver = Driver(pin1=35, pin2=33, pin3=12, pin4=32)

# =============================================================================
# SOCKET SERVER CONFIGURATION
# =============================================================================
# Define the server's IP address and port for receiving control commands.
SERVER_HOST = "192.168.0.177"  # The IP address where this server will run.
SERVER_PORT = 8400             # The port on which to listen for incoming commands.

# =============================================================================
# CAMERA INITIALIZATION
# =============================================================================
# Create a Picamera2 object to control the Raspberry Pi camera.
picam2 = Picamera2()
# Create a video configuration with a resolution of 640x480 pixels.
video_config = picam2.create_video_configuration(main={"size": (640, 480)})
# Apply the configuration and start the camera.
picam2.configure(video_config)
picam2.start()

# =============================================================================
# PID CONTROLLER SETUP
# =============================================================================
# Set a target value (setpoint) for the steering center (position in the image).
setpoint_x = 310
# Initialize the PID controller with specific parameters (Kp, Ki, Kd),
# the setpoint, and limits on the output (correction values between -15 and 15).
pid = PID(0.4, 0.19, 0.017, setpoint=setpoint_x, output_limits=(-15, 15))

# =============================================================================
# SIGNAL HANDLER FOR CLEAN SHUTDOWN
# =============================================================================
def cleanup(signum, frame):
    """
    This function is executed when a termination signal is received.
    It stops the camera, stops the motors, and releases the GPIO pins.
    """
    picam2.stop()    # Stop the camera.
    driver.stop()    # Stop the motors.
    GPIO.cleanup()   # Clean up the GPIO pins.
    print("Camera and GPIOs have been released.")
    exit(0)

# Register the cleanup function for SIGINT (Ctrl+C) and SIGTERM signals.
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# =============================================================================
# FLASK WEB APPLICATION SETUP
# =============================================================================
# Create a Flask application instance to serve the video stream over HTTP.
app = Flask(__name__)

# =============================================================================
# FRAME QUEUE SETUP
# =============================================================================
# Create a queue to hold the latest captured image frame. The queue size is 1.
frame_queue = queue.Queue(maxsize=1)

# Global flag indicating whether the driving algorithm should be active.
driving_enabled = True

# =============================================================================
# FRAME CAPTURE, LINE DETECTION, AND MOTOR CONTROL FUNCTION
# =============================================================================
def capture_frames():
    """
    Continuously captures images from the camera, processes them to detect road lines,
    computes steering corrections using a PID controller, and sends motor commands.
    Also, the processed frame is added to a queue for live streaming.
    """
    while True:
        # Capture an image and convert it from the camera's RGB format to OpenCV's BGR format.
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]

        # --- Define the Region Of Interest (ROI) for the road ---
        # Create an empty mask with the same dimensions as the frame.
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        # Define the four corners of the ROI (polygon covering the road area).
        roi_corners = np.array([[ 
            (0, height),                         # Bottom-left point.
            (width, height),                     # Bottom-right point.
            (int(width * 0.85), int(height * 0.5)),# Top-right point.
            (int(width * 0.08), int(height * 0.65)) # Top-left point.
        ]], dtype=np.int32)
        # Fill the ROI area with white (255) in the mask.
        cv2.fillPoly(roi_mask, roi_corners, 255)
        # Draw the outline of the ROI on the frame in blue for visualization.
        cv2.polylines(frame, roi_corners, isClosed=True, color=(255, 0, 0), thickness=2)

        # --- Process the image within the ROI ---
        # Convert the frame to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply the ROI mask to focus only on the road.
        gray_roi = cv2.bitwise_and(gray, roi_mask)
        # Threshold the image to create a binary mask where white areas indicate potential road lines.
        _, mask_binary = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
        # Find contours (edges) in the binary mask.
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort the contours by area (largest first) and keep up to the two largest.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # Set a "lookahead" point near the bottom of the image (98% of the height) for steering calculation.
        steer_y = int(height * 0.98)

        # --- Determine the steering point based on the detected contours ---
        if len(contours) > 0:
            xs = []
            for cnt in contours:
                if len(cnt) >= 2:
                    # Fit a line to the contour to estimate its direction.
                    vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx = float(vx)
                    vy = float(vy)
                    x0 = float(x0)
                    y0 = float(y0)
                    # Avoid division by zero; if the vertical component is too small, use the base x value.
                    if abs(vy) < 1e-6:
                        x_at_bottom = int(x0)
                    else:
                        # Calculate the x-coordinate of the line at the lookahead point.
                        x_at_bottom = int(((steer_y - y0) * (vx / vy)) + x0)
                    xs.append(x_at_bottom)
            # Compute the steering point:
            if len(xs) == 2:
                # If two lines are detected, take the average.
                cx = int(np.mean(xs))
            elif len(xs) == 1:
                # If only one line is detected, adjust it with an offset for a sharper correction.
                detected_line_x = xs[0]
                offset = 300
                if detected_line_x > width / 2:
                    cx = detected_line_x - offset
                else:
                    cx = detected_line_x + offset
            else:
                cx = setpoint_x  # Default to the setpoint if no lines are detected.
            # Visualize the calculated steering point on the frame.
            cv2.circle(frame, (cx, steer_y), 5, (255, 255, 255), -1)
        else:
            cx = setpoint_x  # Use the default setpoint if no contours are found.

        # --- Calculate the steering correction using the PID controller ---
        correction = pid(cx)

        # **Dynamic Speed Adjustment:**
        # Adjust the base speed based on the magnitude of the correction for safer turns.
        if abs(correction) > 5:
            base_speed = 25
        elif abs(correction) > 10:
            base_speed = 20
        elif abs(correction) > 15:
            base_speed = 10
        else:
            base_speed = 30

        # Calculate individual motor speeds by adding/subtracting the correction.
        left_motor_speed = base_speed + correction
        right_motor_speed = base_speed - correction

        # Limit the motor speeds to the range [0, 40] to prevent excessive speeds.
        left_motor_speed = max(0, min(40, left_motor_speed))
        right_motor_speed = max(0, min(40, right_motor_speed))
        
        if driving_enabled:
            # If driving is enabled, send the calculated speeds to the motor driver.
            # (A debug print statement is available for checking speed values.)
            # print(f"Driving enabled: left_motor_speed={left_motor_speed}, right_motor_speed={right_motor_speed}")
            driver.move(left_motor_speed, right_motor_speed)
        else:
            # If driving is disabled, stop the vehicle.
            driver.stop()

        # Draw the detected contours (in green) and the steering point (in red) for visualization.
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        cv2.circle(frame, (cx, steer_y), 5, (0, 0, 255), -1)

        # Ensure the frame queue holds only the latest frame.
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put(frame)

# =============================================================================
# START BACKGROUND THREAD FOR FRAME CAPTURE
# =============================================================================
# Start a separate thread to run the capture_frames function continuously.
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True  # This thread will automatically close when the main program exits.
capture_thread.start()

# =============================================================================
# SOCKET COMMAND HANDLING
# =============================================================================
def handle_client(conn):
    """
    Process incoming commands over a socket connection.
    Commands can be in JSON format or plain text:
    - "STOP": Deactivates the driving algorithm and stops the vehicle.
    - "START": Reactivates the driving algorithm.
    
    Parameter:
    - conn: The socket connection to the client.
    """
    global driving_enabled
    with conn:
        while True:
            # Read up to 1024 bytes from the connection.
            data = conn.recv(1024)
            if not data:
                break  # Exit the loop if no data is received.
            raw_message = data.decode().strip()  # Decode bytes to a string.
            print(f"Raw message received: {raw_message}")
            try:
                # Attempt to parse the string as JSON.
                payload = json.loads(raw_message)
                command = payload.get("command")
            except json.JSONDecodeError:
                # If parsing fails, treat the message as plain text.
                command = raw_message
            print(f"Processed command: {command}")
            # Process the command.
            if command == "STOP":
                driving_enabled = False  # Disable driving.
                driver.stop()            # Stop the vehicle immediately.
                print("Driving algorithm deactivated.")
            elif command == "START":
                driving_enabled = True   # Enable driving.
                print("Driving algorithm resumed.")
            else:
                print(f"Unknown command received: {command}")

# =============================================================================
# SOCKET SERVER SETUP
# =============================================================================
# Create a TCP socket server to listen for incoming commands.
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind the socket to the specified IP address and port.
server_socket.bind((SERVER_HOST, SERVER_PORT))
# Listen for incoming connections (allowing up to 1 simultaneous connection).
server_socket.listen(1)
print(f"Socket server listening on {SERVER_HOST}:{SERVER_PORT}")

def socket_server():
    """
    Continuously waits for new socket connections.
    For each new connection, a new thread is started to handle the client.
    """
    while True:
        # Accept a new connection.
        conn, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")
        # Start a new daemon thread to handle the client.
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

# Start the socket server in a separate background thread.
threading.Thread(target=socket_server, daemon=True).start()

# =============================================================================
# MJPEG STREAMING SETUP
# =============================================================================
def grab_frames():
    """
    A generator function that continuously retrieves image frames from the queue,
    encodes them as JPEG images, and yields them in a format suitable for MJPEG streaming.
    """
    while True:
        # Retrieve the latest frame from the queue.
        frame = frame_queue.get()
        # Encode the frame as a JPEG image.
        _, jpeg = cv2.imencode('.jpg', frame)
        # Yield the JPEG image along with MJPEG-specific HTTP headers.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# =============================================================================
# FLASK ROUTES FOR VIDEO STREAMING
# =============================================================================
@app.route('/stream.mjpg')
def video_feed():
    """
    Flask route that returns the MJPEG video stream as an HTTP response.
    Clients (like web browsers) can connect to this route to view the live video.
    """
    return Response(grab_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """
    Flask route that returns a simple HTML page.
    This page embeds the MJPEG video stream so that users can view it in their browser.
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

# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    # Run the Flask web server on all network interfaces at port 5000.
    # The 'debug' mode is set to False for production use.
    app.run(host='0.0.0.0', port=5000, debug=False)
