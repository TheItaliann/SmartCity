from ultralytics import YOLO
import threading
from time import sleep

# Load the model
model = YOLO('runs/detect/train66/weights/best.pt')

# Global variable to signal the thread to stop
stop_thread = threading.Event()

def track():
    try:
        # Load and recognize image
        global result
        result = model.track(source='http://192.168.0.166:5000/stream.mjpg',stream=True, show=True, device=0, tracker='bytetrack.yaml')  # Inference
        for r in result:
            boxes = r.boxes
            for box in boxes:
                # Check if box.id is not None
                if box.id is not None:
                    vehicle_id = int(box.id.item())
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
                    xMiddle = (x1 + x2) / 2
                    yMiddle = (y1 + y2) / 2
                    #print(f"Vehicle ID: {vehicle_id}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
                    
                    print(f"Vehicle ID: {vehicle_id}, x: {xMiddle}, y: {yMiddle}")
                else:
                    print("Vehicle ID is None")

            # Check if the stop event is set
            if stop_thread.is_set():
                break
                
    except KeyboardInterrupt:
        print("Stopping")
    finally:
        stop_thread.set()

# Start the tracking in a separate thread
thread = threading.Thread(target=track)
thread.start()

try:
    while thread.is_alive():
        sleep(1)
except KeyboardInterrupt:
    stop_thread.set()
    thread.join()
    print("Thread stopped")
