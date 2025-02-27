# Smart City Projekt - City Bot

## Projektbeschreibung
Dieses Projekt ist eine Simulation einer Smart City mit einem autonomen Fahrzeug (City Bot), das mithilfe eines Raspberry Pi 4B gesteuert wird. Das Fahrzeug verfügt über eine Kamera zur Linienerkennung und navigiert durch eine Stadt, die aktuell eine Größe von 4x4 hat, später jedoch auf 8x8 erweitert werden soll. Zusätzlich gibt es ein Ampelsystem sowie eine digitale Karte zur Positionssteuerung des Fahrzeugs.

## Hauptkomponenten
- **City Bot**: Ein zweirädriges Fahrzeug mit einer Kamera zur Linienerkennung.
- **Raspberry Pi 4B**: Steuereinheit für das Fahrzeug.
- **Kamera**: Erfasst die Umgebung und unterstützt die Linienverfolgung.
- **Kameraüberwachung**: Auf jedem 4x4-Feld befindet sich eine Kamera, die Bilder an einen zentralen Server streamt.
- **KI-Erkennung**: Eine YOLOv11/Ultralytics-KI erkennt das Fahrzeug auf den gestreamten Bildern mit einem eigens trainierten Modell.
- **Digitale Karte**: Ermöglicht die Navigation des Fahrzeugs zu einer bestimmten Position.
- **Ampelsystem**: Die Ampeln senden ihren Status (Grün/Rot), sodass das Fahrzeug entsprechend reagiert.
- **Live-Streaming**: Die Kamerastreams können über einen Webbrowser unter der IP-Adresse und Port 5000 betrachtet werden.

## Geplante Erweiterungen
- Erweiterung der Stadtgröße auf 8x8.
- Verbesserte Objekterkennung und KI-Modelle zur präziseren Fahrzeugerkennung.
- Ausbau des Ampelsystems für eine realistischere Verkehrssteuerung.

## Installation & Nutzung
1. Raspberry Pi 4B mit dem City Bot verbinden und Software installieren.
2. Kameras an den 4x4-Feldern platzieren und mit dem Server verbinden.
3. KI-Modell trainieren und auf den Server hochladen.
4. Ampelsystem mit den jeweiligen Steuerungseinheiten verbinden.
5. Digitale Karte nutzen, um das Fahrzeug zu steuern.
6. Live-Streams über den Webbrowser unter `<IP-Adresse>:5000` abrufen.

## Technologien & Werkzeuge
- **Hardware**: Raspberry Pi 4B, Kamera, Motoren, Sensoren
- **Software**: Python, OpenCV, Flask, YOLOv11/Ultralytics
- **Netzwerk**: IP-basierte Kommunikation, Streaming-Server
- **KI-Training**: Eigene Datensätze für die Fahrzeugerkennung

## Autoren
- [Denis Möller](https://github.com/NinjaV2Kn)
- [Emanuele Alejandro Ustica](https://github.com/TheItaliann)
## Lizenz
Dieses Projekt ist Open-Source und steht unter der MIT-Lizenz.

