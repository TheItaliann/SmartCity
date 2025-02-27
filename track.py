# =============================================================================
# IMPORTS
# =============================================================================
# Import the YOLO model from the ultralytics library for object detection and tracking.
from ultralytics import YOLO
# Import threading to run code in the background (in a separate thread).
import threading
# Import sleep function to pause the program for a short time.
from time import sleep

# =============================================================================
# LOAD THE MODEL
# =============================================================================
# Load the pre-trained YOLO model with the specified weight file.
# This file contains the trained parameters for detecting objects.
model = YOLO('runs/detect/train66/weights/best.pt')

# =============================================================================
# SET UP STOP SIGNAL FOR THREAD
# =============================================================================
# Create a global event that can signal the tracking thread to stop.
stop_thread = threading.Event()

# =============================================================================
# DEFINE THE TRACKING FUNCTION
# =============================================================================
def track():
    """
    This function performs object detection and tracking.
    It connects to a video stream, processes each frame to detect vehicles,
    calculates the center of detected vehicles, and prints the results.
    """
    try:
        # Declare 'result' as a global variable so it can be accessed outside this function if needed.
        global result
        # Use the model to perform tracking on a live video stream from the given URL.
        # 'stream=True' means the model processes a continuous stream.
        # 'show=True' means the video with detections will be displayed.
        # 'device=0' uses the default hardware (like a GPU or CPU).
        # 'tracker='bytetrack.yaml'' specifies the tracking configuration.
        result = model.track(source='http://192.168.0.166:5000/stream.mjpg', stream=True, show=True, device=0, tracker='bytetrack.yaml')  # Inference

        # Loop over each result in the tracking results.
        for r in result:
            # 'boxes' holds all detected objects (vehicles) in the current frame.
            boxes = r.boxes
            # Process each detected box (object).
            for box in boxes:
                # Check if the detected box has an ID assigned (indicating a tracked vehicle).
                if box.id is not None:
                    # Convert the vehicle ID to an integer.
                    vehicle_id = int(box.id.item())
                    # Get the coordinates of the bounding box: top-left (x1, y1) and bottom-right (x2, y2).
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
                    # Calculate the middle of the bounding box (center of the vehicle).
                    xMiddle = (x1 + x2) / 2
                    yMiddle = (y1 + y2) / 2
                    # Print the vehicle ID and its center coordinates.
                    print(f"Vehicle ID: {vehicle_id}, x: {xMiddle}, y: {yMiddle}")
                else:
                    # If the vehicle ID is not available, print a message indicating so.
                    print("Vehicle ID is None")

            # After processing a frame, check if a stop signal has been issued.
            if stop_thread.is_set():
                break  # Exit the loop if the stop signal is set.
                
    except KeyboardInterrupt:
        # If the user interrupts the program (e.g., by pressing Ctrl+C), print a stopping message.
        print("Stopping")
    finally:
        # In any case, signal that the thread should stop.
        stop_thread.set()

# =============================================================================
# START THE TRACKING FUNCTION IN A BACKGROUND THREAD
# =============================================================================
# Create a new thread that will run the track() function.
thread = threading.Thread(target=track)
# Start the thread.
thread.start()

# =============================================================================
# MAIN LOOP TO KEEP THE PROGRAM RUNNING
# =============================================================================
try:
    # While the tracking thread is still running, sleep for 1 second.
    while thread.is_alive():
        sleep(1)
except KeyboardInterrupt:
    # If the user interrupts the main loop (e.g., by pressing Ctrl+C),
    # signal the tracking thread to stop and wait for it to finish.
    stop_thread.set()
    thread.join()
    print("Thread stopped")
