from typing import Tuple
import random

# import os  # TODO remove when finished developing this module

VALID_GRID_CHARS = set(["S", "G", "X", "."])
VALID_ACTIONS = set(["L", "R", "U", "D"])


class InvalidGridError(Exception):
    """
    Raised when:
     - The input grid has more than one 'S' or 'G' cell
     - The input grid has a row with no elements
     - The input grid has an invalid line (not starting with '|' or '-')
     - There are cells with invalid characters (not 'S', 'G', 'X', '.')
     - Rows have different lengths
    """

    pass


class InvalidActionError(Exception):
    """
    Raised when:
     - The input grid has an action that is not 'L', 'R', 'U', 'D'
     - The action taken is not valid for the current state
    """

    pass


class InvalidStateError(Exception):
    """
    Raised when:
     - Trying to set a state that is not within the grid
     - Trying to set a state with an invalid value passed (not tuple, not ints, etc.)
    """

    pass


class Gridworld:
    def __init__(self, input_grid_path: str, input_rules_path: str) -> None:
        self.__default_start_state = None
        self.__current_state = None
        self.__goal_state = None
        self.load_grid(input_grid_path)
        self.load_rules(input_rules_path)
        self.define_dynamics()

    def load_grid(self, grid_path: str) -> None:
        self.grid = []

        with open(grid_path, "r") as grid_file:
            lines = grid_file.readlines()

        s_seen = 0
        g_seen = 0
        col_max = 0
        row = 0
        for line in lines:
            line = line.rstrip()

            if line.startswith("|") and line.endswith("|"):
                self.grid.append([])
                cells = line.split("|")
                cells = cells[1:-1]

                col = 0
                for cell in cells:
                    if cell.upper() not in VALID_GRID_CHARS:
                        raise InvalidGridError("The input grid has cells with invalid characters!")

                    self.grid[row].append(cell)
                    col += 1

                    if cell.upper() == "S":
                        s_seen += 1
                        self.__default_start_state = (row, col)
                    elif cell.upper() == "G":
                        g_seen += 1
                        self.__goal_state = (row, col)

                if col_max != 0:
                    if col != col_max:
                        raise InvalidGridError("The input grid has rows of different lenghts!")
                elif col > 0:
                    col_max = col
                else:
                    raise InvalidGridError("The grid has a row with no elements")

                row += 1

            elif not line.startswith("-"):
                raise InvalidGridError(f"The grid has an invalid line:\n{line}")

        self.row_num = len(self.grid)
        self.column_num = len(self.grid[0])

        if (s_seen != 1) or (g_seen != 1):
            raise InvalidGridError("The input grid has more than one 'S' or 'G' cell!")

    def load_rules(self, rules_path: str) -> None:
        with open(rules_path, "r") as rules_file:
            lines = rules_file.readlines()

        self.actions = set()
        self.custom_rewards = {}
        current_section = None

        for line in lines:
            line = line.rstrip()
            line = line.split("#")[0].strip()  # Remove comments

            if line.startswith("["):
                current_section = line[1:-1]

            elif line != "":
                if current_section == "ACTIONS":
                    if line in VALID_ACTIONS:
                        self.actions.add(line)
                    else:
                        raise InvalidActionError(f"Invalid action: {line}")

                elif current_section == "REWARDS":
                    # TODO implement pattern matching for "DEFAULT = num" and for
                    # "num,num-Action = num", otherwise raise error

                    if line.startswith("DEFAULT"):
                        self.default_reward = int(line.split("=")[1].strip())

                    else:
                        state_action = line.split("=")[0].strip()
                        reward = int(line.split("=")[1].strip())
                        self.custom_rewards[state_action] = reward

                elif current_section == "TRANSITIONS":
                    # TODO implement custom transitions
                    pass

    def add_transitions_to_cell(self, row: int, column: int) -> None:
        for action in self.actions:
            # Check if there is a custom reward for this state-action pair
            try:
                reward = self.custom_rewards[f"{row},{column}-{action}"]
            except KeyError:
                reward = self.default_reward

            # Check what lies ahead
            h_offset = 0
            v_offset = 0
            if action == "L":
                h_offset = -1
            elif action == "R":
                h_offset = 1
            elif action == "U":
                v_offset = -1
            elif action == "D":
                v_offset = 1
            next_col = column + h_offset
            next_row = row + v_offset

            if (0 <= next_row < self.row_num) and (0 <= next_col < self.column_num):
                if self.grid[next_row][next_col] in [".", "S", "G"]:
                    # Add transition to new state
                    self.transitions[row][column][action] = (next_row, next_col, reward)

                elif self.grid[next_row][next_col] == "X":
                    # Moving to an X, stay in same state
                    self.transitions[row][column][action] = (row, column, reward)

            else:
                # Going off grid, stay in same state
                self.transitions[row][column][action] = (row, column, reward)

    def define_dynamics(self) -> None:
        # TODO instead of defining all transitions (memory intensive),
        # could compute them on the fly depending on action selected

        self.transitions = {}

        for row in range(self.row_num):
            for column in range(self.column_num):
                if row not in self.transitions:
                    self.transitions[row] = {}
                if column not in self.transitions[row]:
                    self.transitions[row][column] = {}

                if self.grid[row][column] in [".", "S"]:
                    self.add_transitions_to_cell(row, column)

                else:
                    # No action for 'X' or 'G' states
                    pass

    def print_grid(self) -> None:
        print("-" * (self.column_num * 2 + 1))
        for row in self.grid:
            row_str = "|"
            for cell in row:
                row_str += f"{cell}|"
            print(row_str)
        print("-" * (self.column_num * 2 + 1))

    @property
    def current_state(self):
        return self.__current_state

    @current_state.setter
    def current_state(self, state: Tuple[int, int]):
        row, column = state
        if (0 <= row < self.row_num) and (0 <= column < self.column_num):
            self.__current_state = (row, column)
        else:
            raise InvalidStateError("State not within grid!")

    def initialize(self, method: str = None, state: Tuple[int, int] = None) -> None:
        if method == "default":
            self.current_state = self.__default_start_state

        elif method == "random":
            # Avoid 'X' and 'G'
            row = None
            column = None
            while (row is None) or (column is None) or (self.grid[row][column] not in [".", "S"]):
                row = random.randint(0, self.row_num - 1)
                column = random.randint(0, self.column_num - 1)
            self.current_state = (row, column)

        elif state is not None:
            if (
                (not isinstance(state, tuple))
                or (len(state) != 2)
                or (not isinstance(state[0], int))
                or (not isinstance(state[1], int))
            ):
                raise InvalidStateError("Invalid state passed!")

            # TODO validate state is not an X or G
            self.current_state = state

    def get_possible_actions(self):
        row, column = self.current_state
        return list(self.transitions[row][column].keys())

    def take_action(self, action: str) -> Tuple[int, Tuple[int, int]]:
        # Check if the action is valid
        row, column = self.current_state
        if action in self.transitions[row][column].keys():
            new_row, new_column, reward = self.transitions[row][column][action]
            self.current_state = (new_row, new_column)
            return (reward, (new_row, new_column))
        else:
            raise InvalidActionError(f"Action {action} is not valid for state {row},{column}")


# if __name__ == "__main__":
#     file_path = os.path.dirname(__file__)
#     if file_path != "":
#         os.chdir(file_path)

#     grid_path = "../config/input_grid.txt"
#     rules_path = "../config/grid_rules.config"
#     myGridworld = Gridworld(grid_path, rules_path)
#     myGridworld.print_grid()
