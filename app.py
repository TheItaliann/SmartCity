# =============================================================================
# IMPORTS
# =============================================================================
# Import necessary libraries that provide different functionalities:
from flask import Flask, Response   # Flask helps create a web server and send responses (like web pages and video streams)
from picamera2 import Picamera2      # Picamera2 controls the camera on a Raspberry Pi
import threading                     # threading allows us to run tasks at the same time in the background
import queue                         # queue lets us safely pass data (like images) between threads
import cv2                           # OpenCV (cv2) is used for image processing (like converting colors and drawing shapes)
import signal                        # signal allows us to catch system signals (like when the program is stopped)
import numpy as np                   # NumPy provides support for arrays and mathematical operations
from simple_pid import PID           # simple_pid provides a PID controller (not used directly in this code)

# =============================================================================
# FLASK APPLICATION SETUP
# =============================================================================
# Create a Flask application instance to handle web requests.
app = Flask(__name__)

# =============================================================================
# CAMERA INITIALIZATION
# =============================================================================
# Create a Picamera2 object to control the Raspberry Pi camera.
picam2 = Picamera2()
# Create a video configuration for the camera with a resolution of 640x480 pixels.
video_config = picam2.create_video_configuration(main={"size": (640, 480)})
# Apply the configuration to the camera.
picam2.configure(video_config)
# Start the camera so it begins capturing images.
picam2.start()

# =============================================================================
# FRAME QUEUE SETUP
# =============================================================================
# Create a queue that holds the latest frame. This queue will only store one frame at a time.
frame_queue = queue.Queue(maxsize=1)

# =============================================================================
# SIGNAL HANDLER FOR CLEANUP
# =============================================================================
def cleanup_camera(signum, frame):
    """
    This function is called when the program receives a termination signal (like Ctrl+C).
    It stops the camera and exits the program.
    """
    picam2.stop()  # Stop the camera to release the resource
    print("Kamera freigegeben.")  # Inform that the camera has been released
    exit(0)  # Exit the program

# Register the cleanup function for termination signals (SIGINT and SIGTERM)
signal.signal(signal.SIGINT, cleanup_camera)
signal.signal(signal.SIGTERM, cleanup_camera)

# =============================================================================
# FRAME CAPTURE FUNCTION
# =============================================================================
def capture_frames():
    """
    This function continuously captures frames from the camera,
    processes them (e.g., converts color, applies a mask, and finds a center),
    and then places the processed frame into a queue.
    """
    while True:
        # Capture an image as an array from the camera
        frame = picam2.capture_array()
        # Convert the image from RGB (camera format) to BGR (OpenCV format)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Define lower and upper bounds for a very dark color (almost black)
        low_b = np.array([0, 0, 0], dtype=np.uint8)
        high_b = np.array([5, 5, 5], dtype=np.uint8)
        # Create a mask that only keeps pixels within this dark range
        mask = cv2.inRange(frame, low_b, high_b)
        # Find contours (edges) in the masked image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If any contours are found, identify the largest one and mark its center
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            # Ensure we don't divide by zero if the contour area is zero
            if M["m00"] != 0:
                # Calculate the center of the contour
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # Draw a small white circle at the center of the contour
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
        
        # If the queue is full (already has one frame), remove the old frame
        if frame_queue.full():
            frame_queue.get_nowait()
        # Put the new frame into the queue
        frame_queue.put(frame)

# =============================================================================
# START FRAME CAPTURE IN A BACKGROUND THREAD
# =============================================================================
# Create a new thread to run the capture_frames function so it doesn't block the main program.
capture_thread = threading.Thread(target=capture_frames)
# Set the thread as a daemon so it will close automatically when the main program stops.
capture_thread.daemon = True
capture_thread.start()

# =============================================================================
# MJPEG STREAM GENERATOR
# =============================================================================
def grab_frames():
    """
    This generator function continuously retrieves frames from the queue,
    encodes them as JPEG images, and yields them in a format suitable for MJPEG streaming.
    """
    while True:
        # Get the latest frame from the queue
        frame = frame_queue.get()
        # Encode the frame as a JPEG image
        _, jpeg = cv2.imencode('.jpg', frame)
        # Yield the encoded image with the proper MJPEG formatting
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route('/stream.mjpg')
def video_feed():
    """
    This route returns the MJPEG video stream.
    When a browser accesses this route, it receives a continuous stream of JPEG images.
    """
    return Response(grab_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """
    This is the main HTML page of the web application.
    It displays a simple web page with the video stream embedded in it.
    """
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

# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    # Start the Flask web server so that the video stream can be accessed via a web browser.
    # The server listens on all network interfaces (host '0.0.0.0') on port 5000.
    app.run(host='0.0.0.0', port=5000, debug=False)
