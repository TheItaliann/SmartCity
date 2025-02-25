---

# SmartCity Repository Documentation

The SmartCity project is a multi-device solution for smart city applications. In this repository, different branches are used to deploy code on distinct devices:

- **CityBot:** Code for a mobile robot (CityBot) running on a Raspberry Pi 4B.
- **Top-Camera:** Code for a dedicated camera system running on a Raspberry Pi 4B.
- **DigitalMap:** Code for a digital map interface deployed on an Ubuntu Linux server.

Each branch contains the necessary source code, configurations, and dependency lists tailored to its device.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Branches and Their Devices](#branches-and-their-devices)
   - [CityBot Branch](#citybot-branch)
   - [Top-Camera Branch](#top-camera-branch)
   - [DigitalMap Branch](#digitalmap-branch)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)
8. [Contact](#contact)

---

## 1. Overview

The SmartCity repository is organized into branches that are each deployed on a different device within a smart city ecosystem:

- **CityBot:** This branch provides code for a mobile robot (CityBot) using a Raspberry Pi 4B. It focuses on vehicle control, including motor management (using GPIO and PID control) and command handling.
- **Top-Camera:** This branch is dedicated to a Raspberry Pi 4B configured with a camera. It is designed to capture and stream video (e.g., for surveillance or monitoring).
- **DigitalMap:** This branch runs on an Ubuntu Linux server. It implements a digital mapping interface for visualizing spatial data and potentially managing inputs from the other devices.

---

## 2. Repository Structure

While the exact file layout may vary between branches, each branch generally includes:

- **Source Code:** Device-specific code (for example, camera streaming in Top-Camera, motor control in CityBot, and mapping routines in DigitalMap).
- **Dependencies:** A `requirements.txt` file listing only the packages required for that branch.
- **Configuration Files:** Settings for network communication, hardware interfacing, and other device-specific parameters.
- **Documentation:** This README and any additional docs in a `docs/` directory (if present).

Each branch is self-contained and includes only what is needed to run on its target device.

---

## 3. Branches and Their Devices

### CityBot Branch

**Device:** CityBot (Mobile Robot – Raspberry Pi 4B)

**Description:**  
The **CityBot** branch contains code intended for a mobile robot platform. The robot runs on a Raspberry Pi 4B and is designed for autonomous navigation and remote command handling. Key features typically include:
- **Motor Control:** Using Raspberry Pi GPIO pins with a PID controller to manage motor speeds.
- **Command Handling:** Receiving remote commands (e.g., to start or stop the robot) via socket communication.

**Dependencies:**  
The branch’s `requirements.txt` includes packages such as Flask, picamera2, opencv-python, numpy, RPi.GPIO, simple-pid, and python-snap7.

---

### Top-Camera Branch

**Device:** Raspberry Pi 4B with an Attached Camera

**Description:**  
The **Top-Camera** branch is tailored for a device whose primary function is video capture and streaming. Running on a Raspberry Pi 4B with an attached camera, this branch focuses on:
- **Camera Streaming:** Capturing video using the Raspberry Pi camera interface.
- **Image Processing:** Basic processing routines (e.g., converting color spaces, drawing overlays) using OpenCV.
- **Web Server Interface:** Streaming live video via a Flask-based web application.

**Dependencies:**  
Its `requirements.txt` typically includes Flask, picamera2, opencv-python, and numpy.

---

### DigitalMap Branch

**Device:** Ubuntu Linux Server

**Description:**  
The **DigitalMap** branch is designed to run on an Ubuntu Linux server. It provides a web-based digital mapping interface that can be used to display and manage spatial data from the smart city system. The focus in this branch is on:
- **Mapping Interface:** Displaying maps and spatial data using web technologies.
- **Data Integration:** Collecting and visualizing data from other smart city devices.

**Dependencies:**  
The `requirements.txt` for DigitalMap generally lists Flask, opencv-python, and numpy.

---

## 4. Installation and Setup

### Prerequisites

- **Python 3.10+** installed on the target device.
- A **virtual environment** is recommended for dependency management.
- **Hardware Requirements:**
  - **CityBot and Top-Camera:** Raspberry Pi 4B with appropriate peripherals (camera module, motor controller, etc.).
  - **DigitalMap:** Ubuntu Linux server with web server capabilities.

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/TheItaliann/SmartCity.git
   cd SmartCity
   ```

2. **Switch to the Desired Branch:**
   - For **CityBot**:
     ```bash
     git checkout CityBot
     ```
   - For **Top-Camera**:
     ```bash
     git checkout Top-Camera
     ```
   - For **DigitalMap**:
     ```bash
     git checkout DigitalMap
     ```

3. **Set Up a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Branch-Specific Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Device-Specific Settings:**
   - For **Raspberry Pi** branches: Connect the camera and motor controller as documented. Verify that any IP addresses or GPIO settings in the code match your hardware setup.
   - For **DigitalMap**: Ensure the server has proper network access and any additional mapping configurations as required.

---

## 5. Usage

### Running the Application

- **CityBot Branch:**
  - Run the CityBot application (e.g., `python3 src/lineFollow.py`). The application handles motor control and listens for remote commands.
- **Top-Camera Branch:**
  - Run the top-camera application (e.g., `python3 src/app.py`). The Flask web server will stream live video, which you can view by navigating to:
    ```
    http://<device-ip>:5000/
    ```
- **DigitalMap Branch:**
  - Run the digital map application (e.g., `python3 src/digitalMap.py`). Open a web browser and go to:
    ```
    http://<server-ip>:5000/
    ```
  to access the mapping interface.

### Socket Communication

For branches involving motor control (CityBot), the system listens for TCP socket commands such as:
- **STOP:** To halt the robot.
- **START:** To resume operation.

These commands are sent to the IP address and port specified in the code.

---

## 6. Contributing

Contributions are welcome! To contribute to the SmartCity project:
1. Fork the repository.
2. Create a branch based on the relevant device branch (CityBot, Top-Camera, or DigitalMap).
3. Make your changes and test them on the target device.
4. Open a pull request with a clear explanation of your changes.

For major changes, please open an issue first to discuss your approach.

---

## 7. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 8. Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the repository maintainer via their [GitHub profile](https://github.com/TheItaliann).

---

*This documentation reflects the SmartCity repository as it exists in the CityBot, Top-Camera, and DigitalMap branches. Each branch is tailored for deployment on a specific device, ensuring that the system can be implemented in a multi-device smart city environment.*
