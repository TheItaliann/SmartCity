# Smart City Project - City Bot

## Project Description
This project is a simulation of a Smart City featuring an autonomous vehicle (City Bot) controlled by a Raspberry Pi 4B. The vehicle is equipped with a camera for line detection and navigates through a city currently sized 4x4, with plans to expand to 8x8 in the future. Additionally, the system includes traffic lights and a digital map for vehicle positioning control.

## Main Components
- **City Bot**: A two-wheeled vehicle with a camera for line detection.
- **Raspberry Pi 4B**: The control unit for the vehicle.
- **Camera**: Captures the environment and aids in line tracking.
- **Camera Surveillance**: Each 4x4 field contains a camera that streams images to a central server.
- **AI Recognition**: A YOLOv11/Ultralytics AI detects the vehicle in streamed images using a custom-trained model.
- **Digital Map**: Allows the vehicle to navigate to a specified position.
- **Traffic Light System**: Traffic lights transmit their status (Green/Red), enabling the vehicle to respond accordingly.
- **Live Streaming**: Camera streams can be viewed in a web browser via the IP address and port 5000.

## Planned Enhancements
- Expansion of the city size to 8x8.
- Improved object recognition and AI models for more accurate vehicle detection.
- Extension of the traffic light system for more realistic traffic control.

## Installation & Usage
1. Connect the Raspberry Pi 4B to the City Bot and install the software.
2. Place cameras on the 4x4 fields and connect them to the server.
3. Train the AI model and upload it to the server.
4. Connect the traffic light system to the respective control units.
5. Use the digital map to control the vehicle.
6. Access live streams via a web browser at `<IP-Address>:5000`.

## Technologies & Tools
- **Hardware**: Raspberry Pi 4B, Camera, Motors, Sensors
- **Software**: Python, OpenCV, Flask, YOLOv11/Ultralytics
- **Networking**: IP-based communication, Streaming Server
- **AI Training**: Custom datasets for vehicle detection

## Authors
- [Denis Möller](https://github.com/NinjaV2Kn)
- [Emanuele Alejandro Ustica](https://github.com/TheItaliann)

## License
This project is open-source and licensed under the MIT License.

