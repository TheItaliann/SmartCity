from matplotlib.pyplot import thetagrids
import pygame
import json
import os
from queue import PriorityQueue
import time
import heapq  # Import heapq for binary heap
import socket  # Added for socket communication

pygame.init()

SAVE_FILE = "coordinates.json"
SAVE_FILE_Start = "coords.json"

CURVES = [
    {"name": "unten links", "start": (522, 702), "end": (711, 909)},
    {"name": "unten rechts", "start": (1197, 909), "end": (1386, 702)},
    {"name": "oben rechts", "start": (1386, 198), "end": (1206, 45)},
    {"name": "oben links", "start": (756, 45), "end": (522, 252)},
]


def check_if_path_intersects_curves(path_coords):
    """
    Check if the path goes through any predefined curves and print the curve details.
    
    Args:
        path_coords (list of tuple): The path coordinates as a list of (x, y) tuples.
    
    Returns:
        list of str: Names of curves that the path intersects.
    """
    intersecting_curves = set()  # Use a set to avoid duplicates
    
    for curve in CURVES:
        start = curve["start"]
        end = curve["end"]
        curve_name = curve["name"]

        # Check if any coordinate in the path matches the start or end of the curve
        if any(coord == start or coord == end for coord in path_coords):
            intersecting_curves.add((curve_name, start, end))

    if intersecting_curves:
        for curve_name, start, end in intersecting_curves:
            print(f"The path intersects the curve: {curve_name} at coordinates {start} to {end}")
    else:
        print("The path does not intersect any predefined curves.")
    
    return list(intersecting_curves)


def car_rotation(x1, x2, y1, y2) -> float:
    """Calculate the angle of the car"""
    import math

    # Berechnung des Bewegungsvektors
    dx = x2 - x1
    dy = y2 - y1

    # Berechnung des Winkels in Radiant
    theta = math.atan2(dy, dx)

    # Winkel in Grad konvertieren
    theta_deg = math.degrees(theta)

    # Korrektur des Winkels, um die Basisreferenz anzupassen
    corrected_angle = theta_deg + 90  # Subtrahiere 90 Grad, um die Standardausrichtung anzupassen

    return float(corrected_angle)

def load_image(path):
    try:
        image = pygame.image.load(path)
        return image
    except pygame.error as e:
        print(f"Unable to load image at path: {path}. Error: {e}")
        return None

def load_coords_from_json(filepath):
    """Load coordinates from JSON if the file exists."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print("Error: JSON file is empty or invalid. Using default coordinates.")
            return None
    else:
        print(f"Error: '{filepath}' file not found. Using default coordinates.")
        return None

def update_bot_position(botRect, data, grid, width, height, rows):
    """Update both botRect position in pixels and grid position based on JSON data."""
    if data:
        botRect.topleft = (data.get("x", botRect.x), data.get("y", botRect.y))
        
        start_row, start_col = map_pixel_to_grid(botRect.x, botRect.y, width, height, rows)
        start = grid[start_row][start_col]  
        return start

    return None
# A* Heuristik
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw, map_width, map_height):
    path_coords = []  
    while current in came_from:
        current = came_from[current]
        pixel_x = current.x  
        pixel_y = current.y  
        path_coords.append((pixel_x, pixel_y))  
        current.make_path()
    draw()
    
    # Reverse path to save from start to end
    path_coords.reverse()
    
    
    save_path_to_json(path_coords)
    return path_coords


def save_path_to_json(path_coords):
    """Save the path coordinates to a JSON file."""
    data = {"path": [{"x": x, "y": y} for x, y in path_coords]}
    with open('path_coords.json', 'w') as json_file:
        json.dump(data, json_file)

def algorithm(draw_func, grid, start, end, scan, map_width, map_height):
    count = 0
    open_set = []
    heapq.heappush(open_set, (0, count, start))
    came_from = {}
    g_score = {spot: float('inf') for row in grid for spot in row}
    f_score = {spot: float('inf') for row in grid for spot in row}
    g_score[start] = 0
    f_score[start] = h(start.get_pos(), end.get_pos())
    
    open_set_hash = {start}

    while open_set:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None  

        current = heapq.heappop(open_set)[2]
        open_set_hash.remove(current)

        if current == end:
            path_coords = reconstruct_path(came_from, end, draw_func, map_width, map_height)
            end.make_end()
            return path_coords  

        current.image = scan

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                
                if neighbor not in open_set_hash:
                    count += 1
                    heapq.heappush(open_set, (f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw_func()

        if current != start:
            current.make_closed()
            current.image = (255, 0, 0)  # Mark the scanned path in red

    print("No path found.")
    return None

class Spot:
    def __init__(self, row, col, width, total_rows, barrier, path, data, end=None):
        self.row = row
        self.col = col
        self.x = col * width
        self.y = row * width
        self.width = width
        self.total_rows = total_rows
        self.barrier = barrier
        self.path = path
        self.start = data
        self.end = end if end else path
        self.image = None
        self.neighbors = []

    def get_pos(self):
        return self.row, self.col

    def is_barrier(self):
        return self.image == self.barrier

    def is_path(self):
        return self.image == self.path

    def reset(self):
        self.image = None

    def reset_path(self):
        if self.is_path():
            self.reset()

    def make_barrier(self):
        self.image = self.barrier

    def make_path(self):
        self.image = self.path

    def make_open(self):
        self.image = None  

    def make_closed(self):
        self.image = None  

    def make_end(self):
        self.image = self.end

    def draw(self, screen):
        if isinstance(self.image, pygame.Surface):
            screen.blit(self.image, (self.x, self.y))
        elif isinstance(self.image, tuple):  
            pygame.draw.rect(screen, self.image, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

def make_grid(rows, width, barrier, path, end=None):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([Spot(i, j, gap, rows, barrier, path, end) for j in range(rows)])
    return grid

def draw(screen, grid, rows, width, map_image, bot_image, bot_rect):
    screen.blit(map_image, (0, 0))
    for row in grid:
        for spot in row:
            spot.draw(screen)
    screen.blit(bot_image, bot_rect)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    x, y = pos
    row = min(max(y // gap, 0), rows - 1)
    col = min(max(x // gap, 0), rows - 1)
    return row, col

def reset_grid(grid):
    for row in grid:
        for spot in row:
            spot.reset()

def reset_path(grid, end):
    """Reset the path on the grid, but keep the end point."""
    for row in grid:
        for spot in row:
            if spot.is_path() and spot != end:
                spot.reset()
            if spot.image == (255, 0, 0):  # Reset the scanned path
                spot.reset()

def save_coordinates(end, barriers):
    data = {
        "end": end,
        "barriers": barriers
    }
    with open(SAVE_FILE, 'w') as f:
        json.dump(data, f)

def load_coordinates():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, 'r') as f:
            return json.load(f)
    return None

def map_pixel_to_grid(x, y, grid_width, grid_height, rows):
    """Map pixel coordinates (x, y) to grid coordinates."""
    cell_size = grid_width // rows  # Calculate a uniform cell size for a square grid
    
    # Map pixel (x, y) to grid coordinates (row, col)
    grid_row = int(y / cell_size)
    grid_col = int(x / cell_size)
    
    return grid_row, grid_col

def map_grid_to_pixel(row, col, grid_width, gri, rows):
    """Convert grid coordinates (row, col) back to pixel coordinates for botRect."""
    cell_size = grid_width // rows  # Uniform cell size
    
    # Calculate the top-left pixel of the grid cell (row, col)
    x = col * cell_size
    y = row * cell_size
    
    # Center bot_image within the grid cell by adding half of the cell size
    x_centered = x + cell_size // 2
    y_centered = y + cell_size // 2
    
    return x_centered, y_centered
def check_for_curves(path_coords):
    """Check if the path goes through any predefined curves and print the curve details."""
    for curve in CURVES:
        start = curve["start"]
        end = curve["end"]
        for coord in path_coords:
            if start == coord or end == coord:
                print(f"Curve detected: {curve['name']} from {start} to {end}")

# Example usage after path is found

def send_stop_signal():
    """Send a stop signal via TCP socket to the car."""
    CAR_IP = "192.168.0.177"
    CAR_PORT = 8400  # Ensure the car is listening on this port
    message = "STOP"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((CAR_IP, CAR_PORT))
            sock.sendall(message.encode())
            print("Stop signal sent successfully via socket.")
    except Exception as e:
        print(f"Error sending stop signal via socket: {e}")

def send_start_signal():
    """Send a start  signal via TCP socket to the car."""
    CAR_IP = "192.168.0.177"
    CAR_PORT = 8400  # Ensure the car is listening on this port
    message = "START"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((CAR_IP, CAR_PORT))
            sock.sendall(message.encode())
            print("Start signal sent successfully via socket.")
    except Exception as e:
        print(f"Error sending start signal via socket: {e}")

def check_if_at_destination(botRect, destination_coords):
    bot_x, bot_y = botRect.center  # Use center instead of topleft for more accurate checking
    dest_x, dest_y = destination_coords
    if bot_x == dest_x and bot_y == dest_y:
        send_stop_signal()  # Send stop signal when the destination is reached
        return True
    return False

def load_destination_from_json(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
                if "end" in data:
                    return tuple(data["end"])
        except json.JSONDecodeError:
            print("Error: JSON file is empty or invalid.")
    else:
        print(f"Error: '{filepath}' file not found.")
    return None

# --- Added code: TrafficLightController from sps.py ---
class TrafficLightController:
    def __init__(self, ip, rack, slot):
        self.ip = ip
        self.rack = rack
        self.slot = slot
        import snap7
        from snap7.util import get_bool
        self.snap7 = snap7
        self.get_bool = get_bool
        self.plc = snap7.client.Client()
        try:
            self.plc.connect(self.ip, self.rack, self.slot)
            if not self.plc.get_connected():
                raise ConnectionError("Connection failed.")
            print("SPS connection established.")
        except Exception as e:
            print(f"SPS connection error: {e}")

    def traffic_light(self, light: int):
        try:
            if not self.plc.get_connected():
                print("SPS not connected. Reconnecting...")
                self.plc.connect(self.ip, self.rack, self.slot)
            data = self.plc.read_area(self.snap7.type.Areas.DB, 21, 0, 4)
            lights = [self.get_bool(data, 0, i) for i in range(4)]
            return lights[light]
        except Exception as e:
            print(f"Traffic light error: {e}")
            return None

    def disconnect(self):
        self.plc.disconnect()
        print("SPS disconnected.")
# --- End of added TrafficLightController ---

def main():
    width, height = 1920, 1080  
    rows = 200  
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CityBot Navigation")

    # Colors and images
    barrier = (20, 128, 255)  
    path = (255, 125, 6)   
    end_color = (255, 0, 0)  
    scan = (0, 255, 0) 
    
    barriers = []
    # Load initial data
    data = load_coords_from_json(SAVE_FILE_Start)  # Load bot starting position
    # Create the grid
    grid = make_grid(rows, width, barrier, path, end_color)

    # Load images
    map_image = load_image("images/SmartCityMap.png")
    if map_image is None:
        print("Failed to load the map image.")
        return
    map_image = pygame.transform.scale(map_image, (width, height))

    bot_image = load_image("images/pixelBot.png")
    if bot_image is None:
        print("Failed to load the bot image.")
        return
    bot_image = pygame.transform.scale(bot_image, (135, 120))  
    bot_rect = bot_image.get_rect()  

    # Initialize rotated_bot_image with the original bot_image
    rotated_bot_image = bot_image

    start = update_bot_position(bot_rect, data, grid, width, height, rows)
    if start:
        bot_rect.center = map_grid_to_pixel(start.row, start.col, width, height, rows)

    saved_data = load_coordinates()
    if saved_data:
        end_coords = saved_data["end"]
        barrier_coords = saved_data["barriers"]

        end = grid[end_coords[0]][end_coords[1]]
        end.make_end()

        for barrier in barrier_coords:
            grid[barrier[0]][barrier[1]].make_barrier()  

    previous_position = {"x": bot_rect.centerx, "y": bot_rect.centery}
    running = True
    path_checked = False  # Add a flag to check if the path has been checked for intersections

    # Load destination coordinates from JSON
    destination_coords = load_destination_from_json(SAVE_FILE)

    # Initialize the TrafficLightController and reset traffic check timer
    controller = TrafficLightController('192.168.0.99', 0, 1)
    prev_light_status = None  # Track previous traffic light state
    last_traffic_check = 0  # Timestamp of last traffic light check

    while running:
        data = load_coords_from_json(SAVE_FILE_Start)
        start = update_bot_position(bot_rect, data, grid, width, height, rows)
        if start:
            bot_rect.center = map_grid_to_pixel(start.row, start.col, width, height, rows)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if end:
                    save_coordinates(end.get_pos(), [spot.get_pos() for row in grid for spot in row if spot.is_barrier()])
                running = False
            if event.type == pygame.KEYDOWN:
                print(f"folgende taste wurde gedrückt: {event.key}")
                if event.key == pygame.K_SPACE and start and end:  
                    print("Starting algorithm...")
                    reset_path(grid, end)  # Reset the path before running the algorithm, but keep the end point
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    path_coords = algorithm(lambda: draw(screen, grid, rows, width, map_image, rotated_bot_image, bot_rect), grid, start, end, scan, width, height)
                    if path_coords is None:
                        print("No path found. The end point might be outside the range or blocked.")
                    else:
                        print("Final Path Coordinates (in pixels):", path_coords)
                        check_if_path_intersects_curves(path_coords)
                    path_checked = True  # Set the flag to True after checking the path

                if event.key == pygame.K_c: 
                    end = None
                    barrier.clear()
                    reset_grid(grid)
                    path_checked = False  # Reset the flag when clearing the grid

                if event.key == pygame.K_x:  
                    reset_path(grid, end)
                    path_checked = False  # Reset the flag when resetting the path

        if data:
            current_position = {"x": data["x"], "y": data["y"]}
            if current_position != previous_position:
                angle = car_rotation(previous_position["x"], current_position["x"],
                                     previous_position["y"], current_position["y"])
                rotated_bot_image = pygame.transform.rotate(bot_image, -angle)
                bot_rect = rotated_bot_image.get_rect(center=bot_rect.center)
                previous_position = current_position

            screen.blit(rotated_bot_image, bot_rect)

            if pygame.mouse.get_pressed()[0]:  # Left click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, rows, width)
                spot = grid[row][col]
                
                if not end:
                    end = spot
                    end.make_end() 
                elif spot != end:
                    spot.make_barrier()  
                    barriers.append(spot.get_pos())
                path_checked = False  # Reset the flag when modifying the grid

            if pygame.mouse.get_pressed()[2]:  # Right click to remove barrier or end
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, rows, width)
                spot = grid[row][col]
                if spot:
                    spot.reset()  
                    if spot == end:
                        end = None  
                    if spot.get_pos() in barriers:
                        barriers.remove(spot.get_pos())
                    path_checked = False  # Reset the flag when modifying the grid

        # Repeatedly check traffic light status every 0.5 seconds if car near (1386, 234)
        bot_x, bot_y = bot_rect.center
        now = time.time()
        if abs(bot_x - 1386) < 10 and abs(bot_y - 234) < 10:
            if now - last_traffic_check >= 0.5:
                light_status = controller.traffic_light(2)
                status_str = "Green" if light_status else "Red"
                print(f"Traffic light status at (1386,234): {status_str}")
                if status_str == "Red" and prev_light_status != "Red":
                    send_stop_signal()
                elif status_str == "Green" and prev_light_status == "Red":
                    send_start_signal()
                prev_light_status = status_str
                last_traffic_check = now

        draw(screen, grid, rows, width, map_image, rotated_bot_image, bot_rect)
        
        # Check if the bot has reached the destination
        if destination_coords and check_if_at_destination(bot_rect, destination_coords):
            print("The bot has reached the destination.")
            running = False  # Stop the loop when the destination is reached

        time.sleep(0.1)

    controller.disconnect()
    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        print("Program interrupted and pygame was closed.")