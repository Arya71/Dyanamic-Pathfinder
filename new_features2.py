import pygame
import time
import random
import math
from queue import PriorityQueue
from collections import deque

pygame.init()
pygame.font.init()

# --- Configuration and Constants ---

# Screen and Grid Dimensions
GRID_WIDTH, GRID_HEIGHT = 800, 800
UI_WIDTH = 300
WIDTH, HEIGHT = GRID_WIDTH + UI_WIDTH, GRID_HEIGHT
ROWS = 50
GRID_SIZE = GRID_WIDTH // ROWS

# Window Setup
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced AI Pathfinder Simulation")

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 80, 80)        # Closed Set
GREEN = (120, 220, 120)    # Open Set
BLUE = (100, 100, 255)     # Start Node
YELLOW = (255, 255, 100)   # End Node
PURPLE = (180, 120, 255)   # Path
GREY = (220, 220, 220)     # Grid Lines
UI_BG = (40, 40, 60)
TEXT_COLOR = (230, 230, 230)
HIGHLIGHT_COLOR = (100, 150, 255)
OBSTACLE_COLOR = (80, 80, 90)

# Weighted terrain colors
GRASS_COLOR = (180, 245, 180)   # weight 2
WATER_COLOR = (170, 200, 255)   # weight 5
NORMAL_COLOR = WHITE            # weight 1

# --- Fonts ---
TITLE_FONT = pygame.font.SysFont('corbel', 30, bold=True)
TEXT_FONT = pygame.font.SysFont('corbel', 22)
STATS_FONT = pygame.font.SysFont('consolas', 20)
SMALL_FONT = pygame.font.SysFont('corbel', 18)
WEIGHT_FONT = pygame.font.SysFont('consolas', 14, bold=True)

# --- Obstacle Density Control (1–5 keys) ---
DENSITY_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
density_index = 2  # default 30%

# --- Brush Modes for editing tiles ---
BRUSH_WALL = "wall"
BRUSH_NORMAL = "normal"
BRUSH_GRASS = "grass"
BRUSH_WATER = "water"
brush_mode = BRUSH_WALL  # default: wall brush (matches your old L-click to place walls)


class Node:
    """
    Represents a single node in the grid. It handles its own state,
    color, position, neighbors, and traversal weight.
    """
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * GRID_SIZE
        self.y = row * GRID_SIZE
        self.color = WHITE
        self.neighbors = []
        self.previous = None
        self.is_obstacle_flag = False
        self.weight = 1  # traversal cost to ENTER this node (1=normal)

    def get_pos(self):
        return self.row, self.col

    def is_obstacle(self):
        return self.is_obstacle_flag

    def set_weight(self, w, color):
        self.weight = w
        self.color = color
        self.is_obstacle_flag = False

    def make_normal(self):
        self.set_weight(1, NORMAL_COLOR)

    def make_grass(self):
        self.set_weight(2, GRASS_COLOR)

    def make_water(self):
        self.set_weight(5, WATER_COLOR)

    def reset(self):
        self.previous = None
        self.is_obstacle_flag = False
        self.make_normal()

    def make_start(self):
        # Keep its weight (usually 1). Just change display color.
        self.color = BLUE
    
    def make_end(self):
        self.color = YELLOW

    def make_closed(self):
        if not self.is_obstacle():
            self.color = RED

    def make_open(self):
        if not self.is_obstacle():
            self.color = GREEN

    def make_obstacle(self):
        self.color = OBSTACLE_COLOR
        self.is_obstacle_flag = True
        self.weight = float("inf")  # not used directly, but indicates impassable

    def make_path(self):
        if not self.is_obstacle():
            self.color = PURPLE

    def draw(self, win):
        # Draw main color
        pygame.draw.rect(win, self.color, (self.x, self.y, GRID_SIZE, GRID_SIZE))
        # Draw a slightly inset rect for 3D effect, except for obstacles
        if self.color != OBSTACLE_COLOR and GRID_SIZE > 10:
            pygame.draw.rect(win, GREY, (self.x, self.y, GRID_SIZE, GRID_SIZE), 1)

        # Show weight number if > 1 and not obstacle and tile big enough
        if not self.is_obstacle_flag and self.weight > 1 and GRID_SIZE >= 14:
            label = WEIGHT_FONT.render(str(self.weight), True, BLACK)
            win.blit(label, (self.x + (GRID_SIZE - label.get_width()) // 2,
                             self.y + (GRID_SIZE - label.get_height()) // 2))

    def update_neighbors(self, grid, allow_diagonal):
        """ Populates the neighbors list with valid adjacent nodes. """
        self.neighbors = []
        # Cardinal Directions (Down, Up, Right, Left)
        if self.row < ROWS - 1 and not grid[self.row + 1][self.col].is_obstacle():
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle():
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.col < ROWS - 1 and not grid[self.row][self.col + 1].is_obstacle():
            self.neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle():
            self.neighbors.append(grid[self.row][self.col - 1])

        # Diagonal Directions
        if allow_diagonal:
            if self.row < ROWS - 1 and self.col < ROWS - 1 and not grid[self.row + 1][self.col + 1].is_obstacle():
                self.neighbors.append(grid[self.row + 1][self.col + 1])
            if self.row > 0 and self.col < ROWS - 1 and not grid[self.row - 1][self.col + 1].is_obstacle():
                self.neighbors.append(grid[self.row - 1][self.col + 1])
            if self.row < ROWS - 1 and self.col > 0 and not grid[self.row + 1][self.col - 1].is_obstacle():
                self.neighbors.append(grid[self.row + 1][self.col - 1])
            if self.row > 0 and self.col > 0 and not grid[self.row - 1][self.col - 1].is_obstacle():
                self.neighbors.append(grid[self.row - 1][self.col - 1])


# --- Heuristic Function ---
def h(p1, p2):
    """ Manhattan distance heuristic for A* (admissible with min weight >= 1) """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


# --- Path Reconstruction ---
def reconstruct_path(current, draw, path_length_ref):
    """ Draws the final path by backtracking from the end node. """
    path_length = 0
    while current.previous:
        path_node = current
        prev_node = current.previous
        
        # Draw line connecting centers of nodes
        center1 = (path_node.x + GRID_SIZE // 2, path_node.y + GRID_SIZE // 2)
        center2 = (prev_node.x + GRID_SIZE // 2, prev_node.y + GRID_SIZE // 2)
        pygame.draw.line(win, (255, 255, 255), center1, center2, max(1, GRID_SIZE // 6))

        current.make_path()
        current = current.previous
        path_length += 1
        
    path_length_ref[0] = path_length
    # Redraw to show the lines on top
    draw()


# --- Random Obstacle Generation ---
def generate_random_obstacles(grid, start_node, end_node, density=0.3):
    """
    Randomly turns nodes into obstacles according to 'density' (0.0–1.0).
    Preserves start/end nodes if set. Resets weights to normal for simplicity.
    """
    for row in grid:
        for node in row:
            node.reset()
            if random.random() < density:
                node.make_obstacle()

    # Ensure start & end stay open
    if start_node:
        start_node.reset()
        start_node.make_start()
    if end_node:
        end_node.reset()
        end_node.make_end()


# --- Pathfinding Algorithms ---
# A*, Dijkstra, BFS, DFS; now weighted for A*, Dijkstra, Greedy BFS (g/heuristic).
def generic_search(draw, grid, start, end, algorithm_type, diagonal_allowed=False):
    """
    A generic search function to handle A*, Dijkstra, and Greedy BFS.
    Uses node.weight as the cost to enter each neighbor.
    """
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    
    if algorithm_type == 'astar':
        f_score[start] = h(start.get_pos(), end.get_pos())
    elif algorithm_type == 'dijkstra':
        f_score[start] = 0
    elif algorithm_type == 'greedy':
        f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}
    nodes_explored = 0
    max_space = 0
    start_time = time.time()

    while not open_set.empty():
        pygame.time.delay(30)  # visualization pacing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        current = open_set.get()[2]
        if current in open_set_hash:
            open_set_hash.remove(current)
        nodes_explored += 1

        if current == end:
            end.make_end()
            path_length = [0]
            reconstruct_path(end, draw, path_length)
            start.make_start()
            return {"time": time.time() - start_time, "explored": nodes_explored, "space": max_space, "path_len": path_length[0]}

        cr, cc = current.get_pos()
        for neighbor in current.neighbors:
            nr, nc = neighbor.get_pos()
            # Base step cost is the cost to ENTER the neighbor
            step_cost = neighbor.weight

            # Slightly higher cost for diagonal steps (optional; keeps behavior intuitive)
            if diagonal_allowed and abs(cr - nr) == 1 and abs(cc - nc) == 1:
                step_cost *= math.sqrt(2)

            temp_g_score = g_score[current] + step_cost

            if temp_g_score < g_score[neighbor]:
                neighbor.previous = current
                g_score[neighbor] = temp_g_score
                
                heuristic = h(neighbor.get_pos(), end.get_pos())
                if algorithm_type == 'astar':
                    # heuristic scaled by minimum terrain cost (1) keeps it admissible
                    f_score[neighbor] = temp_g_score + heuristic
                elif algorithm_type == 'dijkstra':
                    f_score[neighbor] = temp_g_score
                elif algorithm_type == 'greedy':
                    f_score[neighbor] = heuristic
                
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        
        max_space = max(max_space, len(open_set_hash))
        draw()

        if current != start:
            current.make_closed()

    return False


def breadth_first_search(draw, grid, start, end):
    """ Breadth-First Search (BFS) Implementation (ignores weights) """
    queue = deque([start])
    visited = {start}
    
    nodes_explored = 0
    max_space = 0
    start_time = time.time()

    while queue:
        pygame.time.delay(30)  # visualization pacing
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); return None
        current = queue.popleft()
        nodes_explored += 1
        if current == end:
            end.make_end()
            path_length = [0]
            reconstruct_path(end, draw, path_length)
            start.make_start()
            return {"time": time.time() - start_time, "explored": nodes_explored, "space": max_space, "path_len": path_length[0]}
        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                neighbor.previous = current
                queue.append(neighbor)
                neighbor.make_open()
        max_space = max(max_space, len(queue))
        draw()
        if current != start: current.make_closed()
    return False


def depth_first_search(draw, grid, start, end):
    """ Depth-First Search (DFS) Implementation (ignores weights) """
    stack = [start]
    visited = {start}
    nodes_explored = 0
    max_space = 0
    start_time = time.time()
    while stack:
        pygame.time.delay(30)  # visualization pacing
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); return None
        current = stack.pop()
        nodes_explored += 1
        if current != start: current.make_closed()
        draw()
        if current == end:
            end.make_end()
            path_length = [0]
            reconstruct_path(end, draw, path_length)
            start.make_start()
            return {"time": time.time() - start_time, "explored": nodes_explored, "space": max_space, "path_len": path_length[0]}
        for neighbor in reversed(current.neighbors):
            if neighbor not in visited:
                visited.add(neighbor)
                neighbor.previous = current
                stack.append(neighbor)
                neighbor.make_open()
        max_space = max(max_space, len(stack))
    return False


# --- Maze Generation (kept from your current version) ---
def generate_maze(grid, start_node, end_node):
    """ Generates a maze using Recursive Backtracking algorithm """
    for row in grid:
        for node in row:
            node.make_obstacle()
    
    def carve_passages(cx, cy, grid_ref):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] # E, S, W, N
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cx + dx*2, cy + dy*2
            if 0 <= nx < ROWS and 0 <= ny < ROWS:
                if grid_ref[nx][ny].is_obstacle():
                    grid_ref[cx + dx][cy].reset()
                    grid_ref[nx][ny].reset()
                    carve_passages(nx, ny, grid_ref)

    # Start carving from a random odd-numbered cell
    start_x, start_y = random.randrange(1, ROWS, 2), random.randrange(1, ROWS, 2)
    grid[start_x][start_y].reset()
    carve_passages(start_x, start_y, grid)
    
    # Ensure start and end nodes are not walls
    if start_node: start_node.reset(); start_node.make_start()
    if end_node: end_node.reset(); end_node.make_end()


# --- Drawing and UI ---

def create_grid():
    """ Initializes the grid with Node objects. """
    return [[Node(i, j) for j in range(ROWS)] for i in range(ROWS)]


def draw_grid_lines(win):
    """ Draws the lines for the grid. """
    for i in range(ROWS + 1):
        pygame.draw.line(win, GREY, (0, i * GRID_SIZE), (GRID_WIDTH, i * GRID_SIZE))
        pygame.draw.line(win, GREY, (i * GRID_SIZE, 0), (i * GRID_SIZE, GRID_HEIGHT))


def draw_ui(win, selected_algo, stats, algorithms, diagonal_allowed):
    """ Draws the entire UI panel on the right side of the screen. """
    ui_rect = pygame.Rect(GRID_WIDTH, 0, UI_WIDTH, HEIGHT)
    pygame.draw.rect(win, UI_BG, ui_rect)

    y_pos = 20
    # Title
    title_text = TITLE_FONT.render("Pathfinding Visualizer", True, WHITE)
    win.blit(title_text, (GRID_WIDTH + (UI_WIDTH - title_text.get_width()) // 2, y_pos))
    y_pos += 50

    # Instructions
    density_pct = int(DENSITY_LEVELS[density_index] * 100)
    instructions = [
        "CONTROLS:",
        "L-Click: Place Start, then End",
        "   ...then paint with Brush",
        "R-Click: Erase to Normal",
        "Space: Run Algorithm",
        "C: Clear Board",
        "M: Generate Maze",
        "R: Random Obstacles",
        f"1-5: Set Density ({density_pct}%)",
        "D: Toggle Diagonals",
        "6-0: Select Algorithm",
        "",
        "BRUSH MODES:",
        "O: Wall (obstacle)",
        "N: Normal (cost 1)",
        "G: Grass (cost 2)",
        "W: Water (cost 5)",
        "",
        "NOTE: A*/Dijkstra use weights.",
        "BFS/DFS ignore weights."
    ]
    for line in instructions:
        line_text = SMALL_FONT.render(line, True, TEXT_COLOR)
        win.blit(line_text, (GRID_WIDTH + 20, y_pos))
        y_pos += 22
    y_pos += 4
    
    # Options
    diag_status = "ON" if diagonal_allowed else "OFF"
    diag_color = GREEN if diagonal_allowed else RED
    diag_text = TEXT_FONT.render(f"Diagonal Movement: ", True, WHITE)
    diag_status_text = TEXT_FONT.render(diag_status, True, diag_color)
    win.blit(diag_text, (GRID_WIDTH + 20, y_pos))
    win.blit(diag_status_text, (GRID_WIDTH + 20 + diag_text.get_width(), y_pos))
    y_pos += 36

    # Density indicator
    dens_text = TEXT_FONT.render(f"Obstacle Density: {density_pct}%", True, WHITE)
    win.blit(dens_text, (GRID_WIDTH + 20, y_pos))
    y_pos += 28

    # Current Brush
    brush_label = {
        BRUSH_WALL: "Wall (Obstacle)",
        BRUSH_NORMAL: "Normal (1)",
        BRUSH_GRASS: "Grass (2)",
        BRUSH_WATER: "Water (5)"
    }[brush_mode]
    brush_text = TEXT_FONT.render(f"Brush: {brush_label}", True, WHITE)
    win.blit(brush_text, (GRID_WIDTH + 20, y_pos))
    y_pos += 36

    # Algorithm Selection
    algo_title = TEXT_FONT.render("ALGORITHMS (6-0):", True, WHITE)
    win.blit(algo_title, (GRID_WIDTH + 20, y_pos))
    y_pos += 30
    
    for i, (name, _) in enumerate(algorithms.items()):
        color = HIGHLIGHT_COLOR if name == selected_algo else TEXT_COLOR
        # Keys mapping shown for clarity
        key_label = ["6", "7", "8", "9", "0"][i]
        key_text = f"{key_label}. {name}"
        line_text = TEXT_FONT.render(key_text, True, color)
        win.blit(line_text, (GRID_WIDTH + 25, y_pos))
        y_pos += 28
    y_pos += 16

    # Stats Display
    stats_title = TEXT_FONT.render("STATISTICS:", True, WHITE)
    win.blit(stats_title, (GRID_WIDTH + 20, y_pos))
    y_pos += 28
    
    if stats:
        if stats == "No Path Found":
            stat_text = STATS_FONT.render(stats, True, RED)
            win.blit(stat_text, (GRID_WIDTH + 25, y_pos))
        else:
            complexity = algorithms[selected_algo]['complexity']
            stat_lines = [
                f"Time Taken   : {stats['time']:.4f} s",
                f"Nodes Explored: {stats['explored']}",
                f"Path Length  : {stats['path_len']}",
                f"Space Used   : {stats['space']} nodes",
                "",
                f"Time Complexity : O({complexity[0]})",
                f"Space Complexity: O({complexity[1]})"
            ]
            for line in stat_lines:
                line_text = STATS_FONT.render(line, True, TEXT_COLOR)
                win.blit(line_text, (GRID_WIDTH + 25, y_pos))
                y_pos += 22


def draw_main(win, grid, selected_algo, stats, algorithms, diagonal_allowed):
    """ Main drawing function, called every frame. """
    win.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw(win)
    draw_grid_lines(win)
    draw_ui(win, selected_algo, stats, algorithms, diagonal_allowed)
    pygame.display.update()


def get_clicked_pos(pos):
    """ Converts mouse coordinates to grid row/col. """
    x, y = pos
    if x >= GRID_WIDTH: return None, None
    return y // GRID_SIZE, x // GRID_SIZE


# --- Main Loop ---
def main(win):
    global density_index, brush_mode
    grid = create_grid()
    start, end = None, None
    
    algorithms = {
        "A* Search": {"func": lambda d, g, s, e, diag=False: generic_search(d, g, s, e, 'astar', diag), "complexity": ("b^d", "b^d")},
        "Dijkstra": {"func": lambda d, g, s, e, diag=False: generic_search(d, g, s, e, 'dijkstra', diag), "complexity": ("V+E log V", "V")},
        "Greedy BFS": {"func": lambda d, g, s, e, diag=False: generic_search(d, g, s, e, 'greedy', diag), "complexity": ("b^d", "b^d")},
        "BFS": {"func": breadth_first_search, "complexity": ("V+E", "V")},
        "DFS": {"func": depth_first_search, "complexity": ("V+E", "V")}
    }
    algo_keys = list(algorithms.keys())
    selected_algorithm_name = algo_keys[0]

    run = True
    started = False
    stats = {}
    diagonal_allowed = False

    while run:
        draw_main(win, grid, selected_algorithm_name, stats, algorithms, diagonal_allowed)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                run = False
            if started: 
                continue

            # Mouse handling
            if pygame.mouse.get_pressed()[0]:  # LEFT CLICK
                row, col = get_clicked_pos(pygame.mouse.get_pos())
                if row is None: 
                    continue
                node = grid[row][col]

                # First two left-clicks set start and end
                if not start and node != end and not node.is_obstacle():
                    start = node
                    start.make_start()
                    start.weight = 1  # ensure start ground cost is 1
                elif not end and node != start and not node.is_obstacle():
                    end = node
                    end.make_end()
                    end.weight = 1
                else:
                    # After start/end are placed, paint with brush
                    if node != start and node != end:
                        if brush_mode == BRUSH_WALL:
                            node.make_obstacle()
                        elif brush_mode == BRUSH_NORMAL:
                            node.make_normal()
                        elif brush_mode == BRUSH_GRASS:
                            node.make_grass()
                        elif brush_mode == BRUSH_WATER:
                            node.make_water()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT CLICK
                row, col = get_clicked_pos(pygame.mouse.get_pos())
                if row is None: 
                    continue
                node = grid[row][col]
                if node == start:
                    node.reset()
                    start = None
                elif node == end:
                    node.reset()
                    end = None
                else:
                    node.make_normal()
            
            # Keyboard handling
            if event.type == pygame.KEYDOWN:
                # Run algorithm
                if event.key == pygame.K_SPACE and start and end:
                    stats = {}
                    for row in grid:
                        for node in row: 
                            node.update_neighbors(grid, diagonal_allowed)
                    started = True
                    func = algorithms[selected_algorithm_name]["func"]
                    if selected_algorithm_name in ("A* Search", "Dijkstra", "Greedy BFS"):
                        result = func(lambda: draw_main(win, grid, selected_algorithm_name, stats, algorithms, diagonal_allowed),
                                      grid, start, end, diagonal_allowed)
                    else:
                        result = func(lambda: draw_main(win, grid, selected_algorithm_name, stats, algorithms, diagonal_allowed),
                                      grid, start, end)
                    stats = "No Path Found" if not result else result
                    started = False

                # Clear board
                if event.key == pygame.K_c:
                    start, end, stats = None, None, {}
                    grid = create_grid()
                
                # Maze generation
                if event.key == pygame.K_m:
                    generate_maze(grid, start, end)

                # Random obstacle generation with current density
                if event.key == pygame.K_r:
                    current_density = DENSITY_LEVELS[density_index]
                    generate_random_obstacles(grid, start, end, current_density)

                # Toggle diagonals
                if event.key == pygame.K_d:
                    diagonal_allowed = not diagonal_allowed
                
                # Obstacle density selection (1–5 => 10%–50%)
                if event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    density_index = event.key - pygame.K_1  # 0..4

                # Algorithm Selection moved to 6,7,8,9,0
                if event.key in (pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0):
                    key_to_index = {
                        pygame.K_6: 0,  # A*
                        pygame.K_7: 1,  # Dijkstra
                        pygame.K_8: 2,  # Greedy BFS
                        pygame.K_9: 3,  # BFS
                        pygame.K_0: 4   # DFS
                    }
                    index = key_to_index[event.key]
                    if index < len(algo_keys):
                        selected_algorithm_name = algo_keys[index]
                        stats = {}

                # Brush selection
                if event.key == pygame.K_o:
                    brush_mode = BRUSH_WALL
                if event.key == pygame.K_n:
                    brush_mode = BRUSH_NORMAL
                if event.key == pygame.K_g:
                    brush_mode = BRUSH_GRASS
                if event.key == pygame.K_w:
                    brush_mode = BRUSH_WATER

    pygame.quit()


if __name__ == "__main__":
    main(win)