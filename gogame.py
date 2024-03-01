import pygame
import numpy as np
import itertools
import sys
import networkx as nx
import collections
from pygame import gfxdraw
import random

# Game constants
BOARD_BROWN = (199, 105, 42)
BOARD_WIDTH = 900
BOARD_BORDER = 75
STONE_RADIUS = 18
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TURN_POS = (BOARD_BORDER, 20)
SCORE_POS = (BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER + 30)
DOT_RADIUS = 4


def make_grid(size):
    """Return list of (start_point, end_point pairs) defining gridlines

    Args:
        size (int): size of grid

    Returns:
        Tuple[List[Tuple[float, float]]]: start and end points for gridlines
    """
    start_points, end_points = [], []

    # vertical start points (constant y)
    xs = np.linspace(BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER, size)
    ys = np.full((size), BOARD_BORDER)
    start_points += list(zip(xs, ys))

    # horizontal start points (constant x)
    xs = np.full((size), BOARD_BORDER)
    ys = np.linspace(BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER, size)
    start_points += list(zip(xs, ys))

    # vertical end points (constant y)
    xs = np.linspace(BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER, size)
    ys = np.full((size), BOARD_WIDTH - BOARD_BORDER)
    end_points += list(zip(xs, ys))

    # horizontal end points (constant x)
    xs = np.full((size), BOARD_WIDTH - BOARD_BORDER)
    ys = np.linspace(BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER, size)
    end_points += list(zip(xs, ys))

    return (start_points, end_points)


def xy_to_colrow(x, y, size):
    """Convert x,y coordinates to column and row number

    Args:
        x (float): x position
        y (float): y position
        size (int): size of grid

    Returns:
        Tuple[int, int]: column and row numbers of intersection
    """
    inc = (BOARD_WIDTH - 2 * BOARD_BORDER) / (size - 1)
    x_dist = x - BOARD_BORDER
    y_dist = y - BOARD_BORDER
    col = int(round(x_dist / inc))
    row = int(round(y_dist / inc))
    return col, row


def colrow_to_xy(col, row, size):
    """Convert column and row numbers to x,y coordinates

    Args:
        col (int): column number (horizontal position)
        row (int): row number (vertical position)
        size (int): size of grid

    Returns:
        Tuple[float, float]: x,y coordinates of intersection
    """
    inc = (BOARD_WIDTH - 2 * BOARD_BORDER) / (size - 1)
    x = int(BOARD_BORDER + col * inc)
    y = int(BOARD_BORDER + row * inc)
    return x, y


def has_no_liberties(board, group):
    """Check if a stone group has any liberties on a given board.

    Args:
        board (object): game board (size * size matrix)
        group (List[Tuple[int, int]]): list of (col,row) pairs defining a stone group

    Returns:
        [boolean]: True if group has any liberties, False otherwise
    """
    for x, y in group:
        if x > 0 and board[x - 1, y] == 0:
            return False
        if y > 0 and board[x, y - 1] == 0:
            return False
        if x < board.shape[0] - 1 and board[x + 1, y] == 0:
            return False
        if y < board.shape[0] - 1 and board[x, y + 1] == 0:
            return False
    return True


def get_stone_groups(board, color):
    """Get stone groups of a given color on a given board

    Args:
        board (object): game board (size * size matrix)
        color (str): name of color to get groups for

    Returns:
        List[List[Tuple[int, int]]]: list of list of (col, row) pairs, each defining a group
    """
    size = board.shape[0]
    color_code = 1 if color == "black" else 2
    xs, ys = np.where(board == color_code)
    graph = nx.grid_graph(dim=[size, size])
    stones = set(zip(xs, ys))
    all_spaces = set(itertools.product(range(size), range(size)))
    stones_to_remove = all_spaces - stones
    graph.remove_nodes_from(stones_to_remove)
    return nx.connected_components(graph)


def is_valid_move(col, row, board):
    """Check if placing a stone at (col, row) is valid on board

    Args:
        col (int): column number
        row (int): row number
        board (object): board grid (size * size matrix)

    Returns:
        boolean: True if move is valid, False otherewise
    """
    # TODO: check for ko situation (infinite back and forth)
    if col < 0 or col >= board.shape[0]:
        return False
    if row < 0 or row >= board.shape[0]:
        return False
    return board[col, row] == 0

def handle_capture(board, color, prisoners):
    """Handle the capture of stones on the board.

    Args:
        board (np.array): The game board.
        color (str): The color of the stones to check for capture.
        prisoners (dict): The prisoners dictionary to update captures.

    Returns:
        bool: True if a capture occurred, False otherwise.
    """
    capture_happened = False
    for group in list(get_stone_groups(board, color)):
        if has_no_liberties(board, group):
            capture_happened = True
            for i, j in group:
                board[i, j] = 0  # Remove the captured stones from the board
            prisoners[color] += len(group)  # Update the prisoners count

    return capture_happened

def handle_invalid_placement(board, col, row, color):
    """Handle the invalid placement of a stone on the board.

    Args:
        board (np.array): The game board.
        col (int): Column number where the stone is placed.
        row (int): Row number where the stone is placed.
        color (str): The color of the stone placed.

    Returns:
        bool: True if the placement is invalid and the stone was removed, False otherwise.
    """
    for group in get_stone_groups(board, color):
        if (col, row) in group:
            break
    if has_no_liberties(board, group):
        board[col, row] = 0  # Remove the invalid stone
        return True  # Invalid placement, stone removed

def simulate_move(self, col, row, self_color, other_color, prisoners):
    invalid_placement = False
    simulated_board = np.copy(self.board)
    # Update board array with the current player's color
    simulated_board[col, row] = 1 if self_color=="black" else 2
    # Simulate capture
    capture_happened = handle_capture(simulated_board, other_color, prisoners)
    if not capture_happened:
        invalid_placement = handle_invalid_placement(simulated_board, col, row, self_color)

    return simulated_board, invalid_placement

# def count_territory():
    


class Game:
    def __init__(self, size):
        self.board = np.zeros((size, size))
        self.size = size
        self.black_turn = True
        self.prisoners = collections.defaultdict(int)
        self.start_points, self.end_points = make_grid(self.size)
        self.running = True
        self.pass_counter = 0  # Add this line to initialize the pass counter
        self.prev_board = None  # Initialize prev_board
        self.undo_count = 0  # Initialize undo_count

    def init_pygame(self):
        pygame.init()
        screen = pygame.display.set_mode((BOARD_WIDTH, BOARD_WIDTH))
        self.screen = screen
        self.ZOINK = pygame.mixer.Sound("wav/zoink.wav")
        self.CLICK = pygame.mixer.Sound("wav/click.wav")
        self.font = pygame.font.SysFont("arial", 30)

    def clear_screen(self):

        # fill board and add gridlines
        self.screen.fill(BOARD_BROWN)
        for start_point, end_point in zip(self.start_points, self.end_points):
            pygame.draw.line(self.screen, BLACK, start_point, end_point)

        # add guide dots
        guide_dots = [3, self.size // 2, self.size - 4]
        for col, row in itertools.product(guide_dots, guide_dots):
            x, y = colrow_to_xy(col, row, self.size)
            gfxdraw.aacircle(self.screen, x, y, DOT_RADIUS, BLACK)
            gfxdraw.filled_circle(self.screen, x, y, DOT_RADIUS, BLACK)

        pygame.display.flip()

    def pass_move(self):
        self.black_turn = not self.black_turn
        self.draw()
        self.pass_counter += 1  # Increment the pass counter on each pass

        # Check if there have been two consecutive passes
        if self.pass_counter >= 2:
            self.end_game()  # Handle the end of the game

    def handle_click(self):
        # Get board position from mouse click
        x, y = pygame.mouse.get_pos()
        col, row = xy_to_colrow(x, y, self.size)
        if not is_valid_move(col, row, self.board):
            self.ZOINK.play()  # Play sound if move is not valid
            return

        # Determine the colors for the current and opponent player
        self_color = "black" if self.black_turn else "white"
        other_color = "white" if self.black_turn else "black"

        simulated_board, invalid_placement = simulate_move(self, col, row, self_color, other_color, self.prisoners)

        #compare the simulated board to the previous board
        if self.prev_board is not None and np.array_equal(simulated_board, self.prev_board):
            self.ZOINK.play()  # Play sound if move is not valid
            return
        else:
            if invalid_placement:
                self.ZOINK.play()  # Play sound if stone placement was invalid
                return  # Exit the function, no need to switch turns or redraw
            # If the move was successful and valid, update the board, switch turns, reset pass counter, and redraw the board
            self.CLICK.play()  # Play sound for valid placement
            self.prev_board = np.copy(self.board)  # Make a deep copy of the board
            self.board = simulated_board
            self.black_turn = not self.black_turn
            self.pass_counter = 0  # Reset the pass counter whenever a move is made
            self.undo_count = 0  # Reset undo_count after a new move is made
            self.draw()  # Redraw the board with the new stone

    def draw(self):
        # draw stones - filled circle and antialiased ring
        self.clear_screen()
        for col, row in zip(*np.where(self.board == 1)):
            x, y = colrow_to_xy(col, row, self.size)
            gfxdraw.aacircle(self.screen, x, y, STONE_RADIUS, BLACK)
            gfxdraw.filled_circle(self.screen, x, y, STONE_RADIUS, BLACK)
        for col, row in zip(*np.where(self.board == 2)):
            x, y = colrow_to_xy(col, row, self.size)
            gfxdraw.aacircle(self.screen, x, y, STONE_RADIUS, WHITE)
            gfxdraw.filled_circle(self.screen, x, y, STONE_RADIUS, WHITE)

        # text for score and turn info
        score_msg = (
            f"Black's Prisoners: {self.prisoners['black']}"
            + f"     White's Prisoners: {self.prisoners['white']}"
        )
        txt = self.font.render(score_msg, True, BLACK)
        self.screen.blit(txt, SCORE_POS)
        turn_msg = (
            f"{'Black' if self.black_turn else 'White'} to move. "
            + "Click to place stone, press P to pass, U to undo."
        )
        txt = self.font.render(turn_msg, True, BLACK)
        self.screen.blit(txt, TURN_POS)

        pygame.display.flip()

    def update(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP:
                self.handle_click()
            elif event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_p:
                    self.pass_move()
                elif event.key == pygame.K_u:
                    if self.prev_board is not None and self.undo_count==0:
                        self.board=self.prev_board
                        self.black_turn = not self.black_turn
                        self.draw()  # Redraw the board with the undone state
                        self.undo_count +=1
                    else:
                        self.ZOINK.play()  # Play sound if move is not valid
                        return

    def end_game(self):
        # Display the final score or a game over message
        print("Game over. Final score - Black's Prisoners: {}, White's Prisoners: {}".format(self.prisoners['black'], self.prisoners['white']))
        self.running = False

if __name__ == "__main__":
    g = Game(size=19)
    g.init_pygame()
    g.clear_screen()
    g.draw()

    while g.running:  # Check if the game is still running
        g.update()
        pygame.time.wait(100)

    pygame.quit()  # Ensure pygame quits after the loop exits
