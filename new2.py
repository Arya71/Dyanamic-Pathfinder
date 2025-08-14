import pygame
import time
import random
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
RED = (255, 80, 80)      # Closed Set
GREEN = (120, 220, 120)  # Open Set
BLUE = (100, 100, 255)   # Start Node
YELLOW = (255, 255, 100) # End Node
PURPLE = (180, 120, 255) # Path
GREY = (220, 220, 220)   # Grid Lines
UI_BG = (40, 40, 60)
TEXT_COLOR = (230, 230, 230)
HIGHLIGHT_COLOR = (100, 150, 255)
OBSTACLE_COLOR = (80, 80, 90)

# --- Fonts ---
TITLE_FONT = pygame.font.SysFont('corbel', 30, bold=True)
TEXT_FONT = pygame.font.SysFont('corbel', 22)
STATS_FONT = pygame.font.SysFont('consolas', 20)
SMALL_FONT = pygame.font.SysFont('corbel', 18)

class Node:
    """
    Represents a single node in the grid. It handles its own state,
    color, position, and neighbors.
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

    def get_pos(self):
        return self.row, self.col

    def is_obstacle(self):
        return self.is_obstacle_flag

    def reset(self):
        self.color = WHITE
        self.previous = None
        self.is_obstacle_flag = False

    def make_start(self):
        self.color = BLUE
    
    def make_end(self):
        self.color = YELLOW

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_obstacle(self):
        self.color = OBSTACLE_COLOR
        self.is_obstacle_flag = True

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        # Draw main color
        pygame.draw.rect(win, self.color, (self.x, self.y, GRID_SIZE, GRID_SIZE))
        # Draw a slightly inset rect for 3D effect, except for obstacles
        if self.color != OBSTACLE_COLOR and GRID_SIZE > 10:
             pygame.draw.rect(win, GREY, (self.x, self.y, GRID_SIZE, GRID_SIZE), 1)

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
    """ Manhattan distance heuristic for A* """
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
        pygame.draw.line(win, (255, 255, 255), center1, center2, GRID_SIZE // 4)

        current.make_path()
        current = current.previous
        path_length += 1
        
    path_length_ref[0] = path_length
    # Redraw to show the lines on top
    draw()


# --- Pathfinding Algorithms ---
# A*, Dijkstra, BFS, DFS functions remain largely the same, but will now
# respect the diagonal movement option passed via the `update_neighbors` method.

def generic_search(draw, grid, start, end, algorithm_type):
    """ A generic search function to handle A*, Dijkstra, and Greedy BFS """
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
        pygame.time.delay(30) # Added delay for visualization
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        current = open_set.get()[2]
        open_set_hash.remove(current)
        nodes_explored += 1

        if current == end:
            end.make_end()
            path_length = [0]
            reconstruct_path(end, draw, path_length)
            start.make_start()
            return {"time": time.time() - start_time, "explored": nodes_explored, "space": max_space, "path_len": path_length[0]}

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                neighbor.previous = current
                g_score[neighbor] = temp_g_score
                
                heuristic = h(neighbor.get_pos(), end.get_pos())
                if algorithm_type == 'astar':
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
    """ Breadth-First Search (BFS) Implementation """
    queue = deque([start])
    visited = {start}
    
    nodes_explored = 0
    max_space = 0
    start_time = time.time()

    while queue:
        pygame.time.delay(30) # Added delay for visualization
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
    """ Depth-First Search (DFS) Implementation """
    stack = [start]
    visited = {start}
    nodes_explored = 0
    max_space = 0
    start_time = time.time()
    while stack:
        pygame.time.delay(30) # Added delay for visualization
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

# --- Maze Generation ---
def generate_maze(grid, start_node, end_node):
    """ Generates a maze using Recursive Backtracking algorithm """
    for row in grid:
        for node in row:
            node.make_obstacle()
    
    def carve_passages(cx, cy, grid):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] # E, S, W, N
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = cx + dx*2, cy + dy*2
            
            if 0 <= nx < ROWS and 0 <= ny < ROWS:
                if grid[nx][ny].is_obstacle():
                    grid[cx + dx][cy].reset()
                    grid[nx][ny].reset()
                    carve_passages(nx, ny, grid)

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
    instructions = [
        "CONTROLS:",
        "L-Click: Place Start/End/Wall",
        "R-Click: Erase Node",
        "Space: Run Algorithm",
        "C: Clear Board",
        "M: Generate Maze",
        "D: Toggle Diagonals",
        "1-5: Select Algorithm"
    ]
    for line in instructions:
        line_text = SMALL_FONT.render(line, True, TEXT_COLOR)
        win.blit(line_text, (GRID_WIDTH + 20, y_pos))
        y_pos += 22
    y_pos += 10
    
    # Options
    diag_status = "ON" if diagonal_allowed else "OFF"
    diag_color = GREEN if diagonal_allowed else RED
    diag_text = TEXT_FONT.render(f"Diagonal Movement: ", True, WHITE)
    diag_status_text = TEXT_FONT.render(diag_status, True, diag_color)
    win.blit(diag_text, (GRID_WIDTH + 20, y_pos))
    win.blit(diag_status_text, (GRID_WIDTH + 20 + diag_text.get_width(), y_pos))
    y_pos += 40

    # Algorithm Selection
    algo_title = TEXT_FONT.render("ALGORITHMS:", True, WHITE)
    win.blit(algo_title, (GRID_WIDTH + 20, y_pos))
    y_pos += 30
    
    for i, (name, _) in enumerate(algorithms.items()):
        color = HIGHLIGHT_COLOR if name == selected_algo else TEXT_COLOR
        key_text = f"{i+1}. {name}"
        line_text = TEXT_FONT.render(key_text, True, color)
        win.blit(line_text, (GRID_WIDTH + 25, y_pos))
        y_pos += 28
    y_pos += 20

    # Stats Display
    stats_title = TEXT_FONT.render("STATISTICS:", True, WHITE)
    win.blit(stats_title, (GRID_WIDTH + 20, y_pos))
    y_pos += 30
    
    if stats:
        if stats == "No Path Found":
            stat_text = STATS_FONT.render(stats, True, RED)
            win.blit(stat_text, (GRID_WIDTH + 25, y_pos))
        else:
            complexity = algorithms[selected_algo]['complexity']
            stat_lines = [
                f"Time Taken  : {stats['time']:.4f} s",
                f"Nodes Explored: {stats['explored']}",
                f"Path Length : {stats['path_len']}",
                f"Space Used  : {stats['space']} nodes",
                "",
                f"Time Complexity: O({complexity[0]})",
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
    grid = create_grid()
    start, end = None, None
    
    algorithms = {
        "A* Search": {"func": lambda d, g, s, e: generic_search(d, g, s, e, 'astar'), "complexity": ("b^d", "b^d")},
        "Dijkstra": {"func": lambda d, g, s, e: generic_search(d, g, s, e, 'dijkstra'), "complexity": ("V+E log V", "V")},
        "Greedy BFS": {"func": lambda d, g, s, e: generic_search(d, g, s, e, 'greedy'), "complexity": ("b^d", "b^d")},
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
            if event.type == pygame.QUIT: run = False
            if started: continue

            if pygame.mouse.get_pressed()[0]:  # LEFT CLICK
                row, col = get_clicked_pos(pygame.mouse.get_pos())
                if row is None: continue
                node = grid[row][col]
                if not start and node != end: start = node; start.make_start()
                elif not end and node != start: end = node; end.make_end()
                elif node != end and node != start: node.make_obstacle()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT CLICK
                row, col = get_clicked_pos(pygame.mouse.get_pos())
                if row is None: continue
                node = grid[row][col]
                node.reset()
                if node == start: start = None
                elif node == end: end = None
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    stats = {}
                    for row in grid:
                        for node in row: node.update_neighbors(grid, diagonal_allowed)
                    started = True
                    result = algorithms[selected_algorithm_name]["func"](lambda: draw_main(win, grid, selected_algorithm_name, stats, algorithms, diagonal_allowed), grid, start, end)
                    stats = "No Path Found" if not result else result
                    started = False

                if event.key == pygame.K_c:
                    start, end, stats = None, None, {}
                    grid = create_grid()
                
                if event.key == pygame.K_m:
                    generate_maze(grid, start, end)

                if event.key == pygame.K_d:
                    diagonal_allowed = not diagonal_allowed
                
                # Algorithm Selection
                if pygame.K_1 <= event.key <= pygame.K_5:
                    index = event.key - pygame.K_1
                    if index < len(algo_keys):
                        selected_algorithm_name = algo_keys[index]
                        stats = {}

    pygame.quit()

if __name__ == "__main__":
    main(win)
