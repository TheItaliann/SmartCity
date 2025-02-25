# =============================================================================
# IMPORTS
# =============================================================================
# Here we bring in various libraries (pre-written code) that help our program do different tasks.
from matplotlib.pyplot import thetagrids  # Not actually used here, but imported from a plotting library
import pygame                             # This helps us create a window, draw images, and handle user input (like a game)
import json                               # For reading from and writing to JSON files (data files)
import os                                 # For working with files and folders on our computer
from queue import PriorityQueue           # For managing a list of items that need to be processed by priority (not used directly)
import time                               # To add pauses in our code and track time
import heapq                              # To efficiently manage a list where we always need the smallest item (used in pathfinding)
import socket                             # To send and receive messages over a network (talking to the car)

# Initialize the pygame library so we can use its features
pygame.init()

# =============================================================================
# GLOBAL CONSTANTS & PREDEFINED DATA
# =============================================================================
# These are fixed values and file names that the program will use.
SAVE_FILE = "coordinates.json"      # File to save where barriers and the endpoint are placed
SAVE_FILE_Start = "coords.json"       # File to save the bot's starting position

# Predefined curves (bends or turns) on the map, with their names and where they start and end
CURVES = [
    {"name": "unten links", "start": (522, 702), "end": (711, 909)},
    {"name": "unten rechts", "start": (1197, 909), "end": (1386, 702)},
    {"name": "oben rechts", "start": (1386, 198), "end": (1206, 45)},
    {"name": "oben links", "start": (756, 45), "end": (522, 252)},
]

# =============================================================================
# UTILITY FUNCTIONS (HELPER FUNCTIONS)
# =============================================================================
# These functions do small, specific tasks to help the rest of the program.

def check_if_path_intersects_curves(path_coords):
    """
    This function checks if the calculated path goes through any of the predefined curves.
    It prints out the details if it does.
    
    Args:
        path_coords (list of tuple): A list of (x, y) positions along the path.
    
    Returns:
        list of tuple: Details of any curves that the path touches.
    """
    intersecting_curves = set()  # Use a set so we don't list the same curve twice
    
    # Go through each curve in our list
    for curve in CURVES:
        start = curve["start"]
        end = curve["end"]
        curve_name = curve["name"]
        # Check if any point on the path exactly matches the start or end of this curve
        if any(coord == start or coord == end for coord in path_coords):
            intersecting_curves.add((curve_name, start, end))
    
    # Tell us what curves are intersected, if any
    if intersecting_curves:
        for curve_name, start, end in intersecting_curves:
            print(f"The path intersects the curve: {curve_name} at coordinates {start} to {end}")
    else:
        print("The path does not intersect any predefined curves.")
    
    return list(intersecting_curves)


def car_rotation(x1, x2, y1, y2) -> float:
    """
    Calculate how much the car should rotate based on its old and new positions.
    
    Args:
        x1, y1: The previous position of the car.
        x2, y2: The current position of the car.
    
    Returns:
        float: The angle (in degrees) to rotate the car image.
    """
    import math
    # Find out how far the car has moved in the x and y directions.
    dx = x2 - x1
    dy = y2 - y1
    # Calculate the angle of movement in radians (a way to measure angles)
    theta = math.atan2(dy, dx)
    # Convert this angle to degrees (a more common measure for people)
    theta_deg = math.degrees(theta)
    # Adjust the angle so that the car image lines up correctly on the screen
    corrected_angle = theta_deg + 90
    return float(corrected_angle)


def load_image(path):
    """
    Try to load an image from a file so we can display it.
    
    Args:
        path (str): The file location of the image.
    
    Returns:
        pygame.Surface or None: The image, or None if it couldn't be loaded.
    """
    try:
        image = pygame.image.load(path)
        return image
    except pygame.error as e:
        print(f"Unable to load image at path: {path}. Error: {e}")
        return None


def load_coords_from_json(filepath):
    """
    Read coordinate data from a JSON file.
    
    Args:
        filepath (str): The location of the JSON file.
    
    Returns:
        dict or None: The data from the file, or None if something goes wrong.
    """
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
    """
    Update where the bot is on the screen and on the grid based on new data.
    
    Args:
        botRect (pygame.Rect): The box that shows where the bot is.
        data (dict): The new position data with x and y values.
        grid (list): The grid made of many small cells.
        width, height (int): The size of the display.
        rows (int): How many cells there are in each row.
    
    Returns:
        Spot: The grid cell where the bot now is.
    """
    if data:
        # Update the bot's position using the x and y values from the data
        botRect.topleft = (data.get("x", botRect.x), data.get("y", botRect.y))
        # Convert the pixel position to a grid cell location
        start_row, start_col = map_pixel_to_grid(botRect.x, botRect.y, width, height, rows)
        start = grid[start_row][start_col]
        return start
    return None

# =============================================================================
# PATH RECONSTRUCTION AND SAVING FUNCTIONS
# =============================================================================
# These functions help build the route (path) the bot will follow and save it to a file.

def h(p1, p2):
    """
    A simple way to estimate distance (Manhattan distance) between two points.
    
    Args:
        p1, p2 (tuple): Two positions (x, y).
    
    Returns:
        int: The distance between the points.
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw, map_width, map_height):
    """
    Work backwards from the endpoint to build the path that was taken.
    
    Args:
        came_from (dict): A map of each spot to the spot before it.
        current: The end spot.
        draw (function): A function that refreshes the display.
        map_width, map_height (int): The size of the map.
    
    Returns:
        list of tuple: A list of (x, y) positions that form the path.
    """
    path_coords = []
    # Go backwards from the end spot to the start spot
    while current in came_from:
        current = came_from[current]
        pixel_x = current.x  # Get the x position of the cell
        pixel_y = current.y  # Get the y position of the cell
        path_coords.append((pixel_x, pixel_y))
        current.make_path()  # Mark this cell as part of the path
    draw()  # Update the display to show the path
    path_coords.reverse()  # Reverse the list so it starts at the beginning
    save_path_to_json(path_coords)  # Save the path to a file
    return path_coords


def save_path_to_json(path_coords):
    """
    Save the path coordinates to a JSON file so we can use them later.
    
    Args:
        path_coords (list of tuple): The list of positions along the path.
    """
    data = {"path": [{"x": x, "y": y} for x, y in path_coords]}
    with open('path_coords.json', 'w') as json_file:
        json.dump(data, json_file)

# =============================================================================
# A* PATHFINDING ALGORITHM
# =============================================================================
def algorithm(draw_func, grid, start, end, scan, map_width, map_height):
    """
    Find the best path from a starting point to an ending point on a grid.
    Think of it as the navigation system for the bot.
    
    Args:
        draw_func (function): Updates what you see on the screen.
        grid (list): A 2D list of cells (Spot objects).
        start (Spot): The cell where the bot starts.
        end (Spot): The target cell.
        scan: A visual indicator (color or image) for checking cells.
        map_width, map_height (int): The size of the map in pixels.
    
    Returns:
        list of tuple: The list of positions (in pixels) that make up the path, or None if no path is found.
    """
    count = 0
    open_set = []  # This will store the cells we need to check next
    heapq.heappush(open_set, (0, count, start))
    came_from = {}  # This dictionary remembers where we came from for each cell
    # Set up two scores for each cell:
    # g_score is the cost from the start to the current cell.
    # f_score is g_score plus an estimate (heuristic) to the end.
    g_score = {spot: float('inf') for row in grid for spot in row}
    f_score = {spot: float('inf') for row in grid for spot in row}
    g_score[start] = 0
    f_score[start] = h(start.get_pos(), end.get_pos())
    open_set_hash = {start}  # A helper set to quickly check if a cell is in open_set

    while open_set:
        # Let the program handle window closing events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        # Take the cell with the lowest estimated cost
        current = heapq.heappop(open_set)[2]
        open_set_hash.remove(current)

        # If we reached the end, rebuild and return the path
        if current == end:
            path_coords = reconstruct_path(came_from, end, draw_func, map_width, map_height)
            end.make_end()  # Mark the end cell visually
            return path_coords

        # Mark the current cell as scanned (using the given color or image)
        current.image = scan

        # Look at each neighbor (adjacent cell)
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1  # Assume each move costs the same
            if temp_g_score < g_score[neighbor]:
                # This is a better path to the neighbor, so record it
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    heapq.heappush(open_set, (f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()  # Mark the neighbor as open (to be checked)
        draw_func()  # Refresh the screen

        if current != start:
            current.make_closed()  # Mark the cell as already checked
            current.image = (255, 0, 0)  # Color it red to show it was scanned

    print("No path found.")
    return None

# =============================================================================
# SPOT CLASS AND GRID CREATION
# =============================================================================
class Spot:
    """
    This class represents one small square (cell) in the grid.
    Think of it as one "tile" on the map.
    """
    def __init__(self, row, col, width, total_rows, barrier, path, data, end=None):
        self.row = row
        self.col = col
        self.x = col * width   # The x position on the screen
        self.y = row * width   # The y position on the screen
        self.width = width     # How big this cell is
        self.total_rows = total_rows
        self.barrier = barrier  # What the cell looks like when it's a barrier (obstacle)
        self.path = path        # What the cell looks like when it is part of the chosen path
        self.start = data       # What the cell looks like when it's the starting point
        self.end = end if end else path  # What the cell looks like when it's the end point
        self.image = None       # The current visual appearance of the cell
        self.neighbors = []     # A list of adjacent cells (neighbors)

    def get_pos(self):
        """Return the position of the cell in the grid (row, column)."""
        return self.row, self.col

    def is_barrier(self):
        """Return True if this cell is an obstacle."""
        return self.image == self.barrier

    def is_path(self):
        """Return True if this cell is marked as part of the path."""
        return self.image == self.path

    def reset(self):
        """Clear any markings on this cell."""
        self.image = None

    def reset_path(self):
        """Clear the path marking from this cell."""
        if self.is_path():
            self.reset()

    def make_barrier(self):
        """Mark this cell as an obstacle."""
        self.image = self.barrier

    def make_path(self):
        """Mark this cell as part of the path."""
        self.image = self.path

    def make_open(self):
        """Mark this cell as open for checking (no obstacle)."""
        self.image = None

    def make_closed(self):
        """Mark this cell as already checked."""
        self.image = None

    def make_end(self):
        """Mark this cell as the destination."""
        self.image = self.end

    def draw(self, screen):
        """
        Draw this cell on the screen.
        
        Args:
            screen (pygame.Surface): The window where everything is shown.
        """
        if isinstance(self.image, pygame.Surface):
            screen.blit(self.image, (self.x, self.y))
        elif isinstance(self.image, tuple):
            pygame.draw.rect(screen, self.image, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        """
        Look at the four directions (up, down, left, right) and record neighboring cells that are not obstacles.
        
        Args:
            grid (list): The full grid of cells.
        """
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
        # This function is needed for the priority queue but isn't used for actual comparisons.
        return False


def make_grid(rows, width, barrier, path, end=None):
    """
    Create a grid (a list of lists) filled with Spot objects.
    
    Args:
        rows (int): How many cells there are in each row (and column).
        width (int): Total width of the grid in pixels.
        barrier: How a barrier cell is represented.
        path: How a path cell is represented.
        end: (Optional) How the end cell is represented.
    
    Returns:
        list: A 2D list (grid) of Spot objects.
    """
    grid = []
    gap = width // rows  # Determine the size of each cell
    for i in range(rows):
        grid.append([Spot(i, j, gap, rows, barrier, path, end) for j in range(rows)])
    return grid

# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================
def draw(screen, grid, rows, width, map_image, bot_image, bot_rect):
    """
    Draw everything on the screen: the background map, the grid, and the bot.
    
    Args:
        screen (pygame.Surface): The window.
        grid (list): The 2D grid of cells.
        rows (int): Number of rows in the grid.
        width (int): The window's width in pixels.
        map_image (pygame.Surface): The background image (map).
        bot_image (pygame.Surface): The image of the bot.
        bot_rect (pygame.Rect): The position of the bot.
    """
    screen.blit(map_image, (0, 0))  # Draw the background map
    for row in grid:
        for spot in row:
            spot.draw(screen)         # Draw each cell
    screen.blit(bot_image, bot_rect)  # Draw the bot on top of everything
    pygame.display.update()           # Refresh the display


def get_clicked_pos(pos, rows, width):
    """
    Convert a mouse click (in pixels) to a grid cell position.
    
    Args:
        pos (tuple): The (x, y) pixel coordinates of the mouse click.
        rows (int): Number of rows in the grid.
        width (int): The width of the grid in pixels.
    
    Returns:
        tuple: The (row, col) of the grid cell that was clicked.
    """
    gap = width // rows
    x, y = pos
    row = min(max(y // gap, 0), rows - 1)
    col = min(max(x // gap, 0), rows - 1)
    return row, col


def reset_grid(grid):
    """
    Clear all markings from the grid, resetting every cell.
    
    Args:
        grid (list): The 2D grid of cells.
    """
    for row in grid:
        for spot in row:
            spot.reset()


def reset_path(grid, end):
    """
    Remove the path markings from the grid but keep the end cell marked.
    
    Args:
        grid (list): The 2D grid of cells.
        end (Spot): The destination cell.
    """
    for row in grid:
        for spot in row:
            if spot.is_path() and spot != end:
                spot.reset()
            if spot.image == (255, 0, 0):  # Also remove red markings from scanned cells
                spot.reset()


def save_coordinates(end, barriers):
    """
    Save the position of the endpoint and all barriers to a file.
    
    Args:
        end (tuple): The grid position of the end.
        barriers (list): A list of grid positions for obstacles.
    """
    data = {
        "end": end,
        "barriers": barriers
    }
    with open(SAVE_FILE, 'w') as f:
        json.dump(data, f)


def load_coordinates():
    """
    Load saved endpoint and barrier data from a file.
    
    Returns:
        dict or None: The loaded data, or None if the file doesn't exist.
    """
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, 'r') as f:
            return json.load(f)
    return None


def map_pixel_to_grid(x, y, grid_width, grid_height, rows):
    """
    Convert pixel coordinates to grid coordinates.
    
    Args:
        x, y (int): Pixel positions.
        grid_width, grid_height (int): The size of the grid in pixels.
        rows (int): Number of rows in the grid.
    
    Returns:
        tuple: (row, col) on the grid.
    """
    cell_size = grid_width // rows
    grid_row = int(y / cell_size)
    grid_col = int(x / cell_size)
    return grid_row, grid_col


def map_grid_to_pixel(row, col, grid_width, gri, rows):
    """
    Convert grid coordinates back to pixel coordinates so we can position the bot.
    
    Args:
        row, col (int): The cell position in the grid.
        grid_width (int): The width of the grid.
        gri: (Not used, might be a typo)
        rows (int): Number of rows in the grid.
    
    Returns:
        tuple: (x, y) pixel coordinates at the center of the cell.
    """
    cell_size = grid_width // rows
    x = col * cell_size
    y = row * cell_size
    x_centered = x + cell_size // 2
    y_centered = y + cell_size // 2
    return x_centered, y_centered


def check_for_curves(path_coords):
    """
    Check if any part of the path goes through the curves defined at the start.
    
    Args:
        path_coords (list): A list of (x, y) positions making up the path.
    """
    for curve in CURVES:
        start = curve["start"]
        end = curve["end"]
        for coord in path_coords:
            if start == coord or end == coord:
                print(f"Curve detected: {curve['name']} from {start} to {end}")

# =============================================================================
# SIGNAL FUNCTIONS FOR CAR CONTROL
# =============================================================================
def send_stop_signal():
    """
    Connect to the car over the network and send a "STOP" command.
    """
    CAR_IP = "192.168.0.177"
    CAR_PORT = 8400  # The port where the car listens for commands
    message = "STOP"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((CAR_IP, CAR_PORT))
            sock.sendall(message.encode())
            print("Stop signal sent successfully via socket.")
    except Exception as e:
        print(f"Error sending stop signal via socket: {e}")


def send_start_signal():
    """
    Connect to the car over the network and send a "START" command.
    """
    CAR_IP = "192.168.0.177"
    CAR_PORT = 8400
    message = "START"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((CAR_IP, CAR_PORT))
            sock.sendall(message.encode())
            print("Start signal sent successfully via socket.")
    except Exception as e:
        print(f"Error sending start signal via socket: {e}")


def check_if_at_destination(botRect, destination_coords):
    """
    Check if the bot has reached the destination.
    
    Args:
        botRect (pygame.Rect): The bot's current position.
        destination_coords (tuple): The destination's (x, y) pixel coordinates.
    
    Returns:
        bool: True if the bot is at the destination, otherwise False.
    """
    bot_x, bot_y = botRect.center  # Use the center of the bot for accuracy
    dest_x, dest_y = destination_coords
    if bot_x == dest_x and bot_y == dest_y:
        send_stop_signal()  # Tell the car to stop when the destination is reached
        return True
    return False


def load_destination_from_json(filepath):
    """
    Load the destination coordinates from a JSON file.
    
    Args:
        filepath (str): The file path for the JSON file.
    
    Returns:
        tuple or None: The destination coordinates, or None if not available.
    """
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

# =============================================================================
# TRAFFIC LIGHT CONTROLLER (PLC COMMUNICATION)
# =============================================================================
class TrafficLightController:
    def __init__(self, ip, rack, slot):
        """
        Set up communication with a traffic light controller (PLC).
        
        Args:
            ip (str): The IP address of the PLC.
            rack (int): The rack number.
            slot (int): The slot number.
        """
        self.ip = ip
        self.rack = rack
        self.slot = slot
        import snap7  # Library to communicate with the PLC
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
        """
        Check the status of a specific traffic light.
        
        Args:
            light (int): The index of the traffic light.
        
        Returns:
            bool or None: True if the light is green, False if red, or None if an error occurs.
        """
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
        """Disconnect from the traffic light controller."""
        self.plc.disconnect()
        print("SPS disconnected.")

# =============================================================================
# MAIN APPLICATION FUNCTION
# =============================================================================
def main():
    """
    This is the main function that runs the CityBot Navigation program.
    It sets up the window, loads images and grid data, processes user inputs,
    runs the pathfinding algorithm, and communicates with the car and traffic lights.
    """
    width, height = 1920, 1080  # Size of the display window
    rows = 200                # How many cells there are in our grid (both rows and columns)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CityBot Navigation")

    # Define colors (or images) for different grid elements:
    barrier = (20, 128, 255)  # Color used for obstacles
    path = (255, 125, 6)      # Color used for the chosen path
    end_color = (255, 0, 0)   # Color used for the destination cell
    scan = (0, 255, 0)        # Color used to show cells that have been checked

    barriers = []  # List to store where obstacles are placed

    # Load the bot's starting position from a file
    data = load_coords_from_json(SAVE_FILE_Start)
    # Create the grid of cells
    grid = make_grid(rows, width, barrier, path, end_color)

    # Load the background map image
    map_image = load_image("images/SmartCityMap.png")
    if map_image is None:
        print("Failed to load the map image.")
        return
    map_image = pygame.transform.scale(map_image, (width, height))

    # Load the image of the bot
    bot_image = load_image("images/pixelBot.png")
    if bot_image is None:
        print("Failed to load the bot image.")
        return
    bot_image = pygame.transform.scale(bot_image, (135, 120))
    bot_rect = bot_image.get_rect()

    # Keep a rotated version of the bot image (so we can show its direction)
    rotated_bot_image = bot_image

    # Update the bot's position based on the loaded data
    start = update_bot_position(bot_rect, data, grid, width, height, rows)
    if start:
        bot_rect.center = map_grid_to_pixel(start.row, start.col, width, height, rows)

    # Load saved grid data (destination and obstacles) from a file
    saved_data = load_coordinates()
    if saved_data:
        end_coords = saved_data["end"]
        barrier_coords = saved_data["barriers"]
        end = grid[end_coords[0]][end_coords[1]]
        end.make_end()  # Mark the destination cell
        for barrier in barrier_coords:
            grid[barrier[0]][barrier[1]].make_barrier()

    previous_position = {"x": bot_rect.centerx, "y": bot_rect.centery}
    running = True
    path_checked = False  # This flag indicates if the path has been checked for curves

    # Load the destination coordinates (where the bot should eventually go) from a file
    destination_coords = load_destination_from_json(SAVE_FILE)

    # Set up communication with the traffic light controller
    controller = TrafficLightController('192.168.0.99', 0, 1)
    prev_light_status = None  # Remember what the traffic light was showing last time
    last_traffic_check = 0    # Time of the last traffic light check

    # =============================================================================
    # MAIN LOOP: This loop runs continuously until the program is closed.
    # =============================================================================
    while running:
        data = load_coords_from_json(SAVE_FILE_Start)
        start = update_bot_position(bot_rect, data, grid, width, height, rows)
        if start:
            bot_rect.center = map_grid_to_pixel(start.row, start.col, width, height, rows)
        
        # Process any events (like key presses or closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if end:
                    # Save the current grid (destination and obstacles) before closing
                    save_coordinates(end.get_pos(), [spot.get_pos() for row in grid for spot in row if spot.is_barrier()])
                running = False
            if event.type == pygame.KEYDOWN:
                print(f"Key pressed: {event.key}")
                if event.key == pygame.K_SPACE and start and end:
                    print("Starting algorithm...")
                    reset_path(grid, end)  # Clear any old path markings but keep the destination
                    # Update the neighbor information for each cell in the grid
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    # Run the A* algorithm to find the best path from start to destination
                    path_coords = algorithm(lambda: draw(screen, grid, rows, width, map_image, rotated_bot_image, bot_rect),
                                            grid, start, end, scan, width, height)
                    if path_coords is None:
                        print("No path found. The end point might be outside the range or blocked.")
                    else:
                        print("Final Path Coordinates (in pixels):", path_coords)
                        check_if_path_intersects_curves(path_coords)
                    path_checked = True
                if event.key == pygame.K_c:
                    # Clear the destination and obstacles if the user presses 'c'
                    end = None
                    barrier.clear()
                    reset_grid(grid)
                    path_checked = False
                if event.key == pygame.K_x:
                    # Reset the current path if the user presses 'x'
                    reset_path(grid, end)
                    path_checked = False

        # Update the bot's rotation and position if new position data is available
        if data:
            current_position = {"x": data["x"], "y": data["y"]}
            if current_position != previous_position:
                # Calculate how much the bot should rotate based on its movement
                angle = car_rotation(previous_position["x"], current_position["x"],
                                     previous_position["y"], current_position["y"])
                rotated_bot_image = pygame.transform.rotate(bot_image, -angle)
                bot_rect = rotated_bot_image.get_rect(center=bot_rect.center)
                previous_position = current_position
            screen.blit(rotated_bot_image, bot_rect)

            # If the user left-clicks, use the mouse position to set a destination or add obstacles
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, rows, width)
                spot = grid[row][col]
                if not end:
                    end = spot
                    end.make_end()
                elif spot != end:
                    spot.make_barrier()
                    barriers.append(spot.get_pos())
                path_checked = False
            # If the user right-clicks, remove an obstacle or clear the destination
            if pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, rows, width)
                spot = grid[row][col]
                if spot:
                    spot.reset()
                    if spot == end:
                        end = None
                    if spot.get_pos() in barriers:
                        barriers.remove(spot.get_pos())
                    path_checked = False

        # =============================================================================
        # TRAFFIC LIGHT CHECK: When the bot is near a specific location, check the traffic light.
        # =============================================================================
        bot_x, bot_y = bot_rect.center
        now = time.time()
        # If the bot is very close to (1386, 234), then:
        if abs(bot_x - 1386) < 10 and abs(bot_y - 234) < 10:
            if now - last_traffic_check >= 0.5:  # Check every half a second
                light_status = controller.traffic_light(2)
                status_str = "Green" if light_status else "Red"
                print(f"Traffic light status at (1386,234): {status_str}")
                # If the light is red and it wasn't red before, send a stop command to the car
                if status_str == "Red" and prev_light_status != "Red":
                    send_stop_signal()
                # If the light is green and it was red before, send a start command
                elif status_str == "Green" and prev_light_status == "Red":
                    send_start_signal()
                prev_light_status = status_str
                last_traffic_check = now

        draw(screen, grid, rows, width, map_image, rotated_bot_image, bot_rect)
        
        # Check if the bot has reached the final destination; if so, stop the program
        if destination_coords and check_if_at_destination(bot_rect, destination_coords):
            print("The bot has reached the destination.")
            running = False

        time.sleep(0.1)  # A short pause to control the loop speed

    controller.disconnect()
    pygame.quit()

# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        print("Program interrupted and pygame was closed.")
