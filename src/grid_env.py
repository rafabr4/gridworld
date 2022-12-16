from typing import Tuple
import random
import re

# import os  # TODO remove when finished developing this module


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


class InvalidRewardConfigError(Exception):
    """
    Raised when:
     - A reward in the config file doesn't match the default reward format
     - A reward in the config file doesn't match the custom reward format
     - A reward in the config file specifies a state that is off grid
    """

    pass


class IlegalCellChangeError(Exception):
    """
    Raised when trying to change the current cell from outside the class
    """

    pass


class IlegalStateChangeError(Exception):
    """
    Raised when trying to change the current state from outside the class
    """

    pass


class Gridworld:
    _VALID_GRID_CHARS = set(["S", "G", "X", "."])
    _VALID_ACTIONS = set(["L", "R", "U", "D"])
    _PATTERN_DEFAULT_REWARD = r"DEFAULT = -{0,1}\d+"
    _ACTIONS_FOR_REGEX = "[" + "".join(_VALID_ACTIONS) + "]"
    _PATTERN_CUSTOM_REWARD = r"\d+,\d+-" + _ACTIONS_FOR_REGEX + r" = -{0,1}\d+"

    def __init__(self, input_grid_path: str, input_rules_path: str) -> None:
        self._default_start_state = None
        self._current_state = None
        self._current_cell = None
        self._goal_state = None
        self._load_grid(input_grid_path)
        self._load_rules(input_rules_path)
        self._define_dynamics()
        self._validate_custom_rewards()

    def _load_grid(self, grid_path: str) -> None:
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
                    if cell.upper() not in self._VALID_GRID_CHARS:
                        raise InvalidGridError("The input grid has cells with invalid characters!")

                    if cell.upper() == "S":
                        s_seen += 1
                        self._default_start_state = (row, col)
                    elif cell.upper() == "G":
                        g_seen += 1
                        self._goal_state = (row, col)

                    self.grid[row].append(cell)
                    col += 1

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
            raise InvalidGridError("The input grid has invalid amount of 'S' or 'G' cell!")

    def _load_rules(self, rules_path: str) -> None:
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
                    if line in self._VALID_ACTIONS:
                        self.actions.add(line)
                    else:
                        raise InvalidActionError(f"Invalid action: {line}")

                elif current_section == "REWARDS":
                    if re.match(self._PATTERN_DEFAULT_REWARD, line):
                        self.default_reward = int(line.split("=")[1].strip())

                    elif re.match(self._PATTERN_CUSTOM_REWARD, line):
                        state_action = line.split("=")[0].strip()
                        reward = int(line.split("=")[1].strip())
                        self.custom_rewards[state_action] = reward

                    else:
                        raise InvalidRewardConfigError(f"Invalid reward: {line}")

                elif current_section == "TRANSITIONS":
                    # TODO implement custom transitions
                    pass

    def _add_transitions_to_cell(self, row: int, column: int) -> None:
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

    def _define_dynamics(self) -> None:
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
                    self._add_transitions_to_cell(row, column)

                else:
                    # No action for 'X' or 'G' states
                    pass

    def _validate_custom_rewards(self) -> None:
        for state_action in self.custom_rewards.keys():
            # No need to validate action because it was already done in the regex matching
            state, _ = state_action.split("-")
            row, column = [int(x) for x in state.split(",")]

            if (row < 0) or (row >= self.row_num) or (column < 0) or (column >= self.column_num):
                raise InvalidRewardConfigError(
                    f"Invalid reward due to state off grid: {state_action}"
                )

    def print_grid(self) -> None:
        print("-" * (self.column_num * 2 + 1))
        for row in self.grid:
            row_str = "|"
            for cell in row:
                row_str += f"{cell}|"
            print(row_str)
        print("-" * (self.column_num * 2 + 1))

    def _change_state_and_cell(self, state: Tuple[int, int]) -> None:
        self._current_state = state
        self._current_cell = self.grid[state[0]][state[1]]

    def initialize(self, method: str = None, state: Tuple[int, int] = None) -> None:
        if method == "default":
            self._change_state_and_cell(self._default_start_state)

        elif method == "random":
            # Avoid 'X' and 'G'
            row = None
            column = None
            while (row is None) or (column is None) or (self.grid[row][column] not in [".", "S"]):
                row = random.randint(0, self.row_num - 1)
                column = random.randint(0, self.column_num - 1)
            self._change_state_and_cell((row, column))

        elif state is not None:
            # Check it has the format: Tuple[int, int]
            if (
                (not isinstance(state, tuple))
                or (len(state) != 2)
                or (not isinstance(state[0], int))
                or (not isinstance(state[1], int))
            ):
                raise InvalidStateError("Invalid state passed!")

            # Check state is within grid and is not an 'X' or 'G'
            if (0 <= state[0] < self.row_num) and (0 <= state[1] < self.column_num):
                if self.grid[state[0]][state[1]] in [".", "S"]:
                    self._change_state_and_cell(state)
                else:
                    raise InvalidStateError("State needs to be a cell with '.' or 'S'")
            else:
                raise InvalidStateError("State not within grid!")

    def get_possible_actions(self):
        row, column = self.current_state
        return list(self.transitions[row][column].keys())

    def take_action(self, action: str) -> Tuple[int, Tuple[int, int]]:
        # Check if the action is valid
        row, column = self.current_state
        if action in self.transitions[row][column].keys():
            new_row, new_column, reward = self.transitions[row][column][action]
            self._change_state_and_cell((new_row, new_column))
            # TODO if landed in 'G', do something
            return (reward, (new_row, new_column))
        else:
            raise InvalidActionError(f"Action {action} is not valid for state {row},{column}")

    @property
    def current_cell(self):
        return self._current_cell

    @current_cell.setter
    def current_cell(self, value):
        raise IlegalCellChangeError("Cannot change current cell from outside the class!")

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, value):
        raise IlegalStateChangeError("Cannot change current state from outside the class!")


# if __name__ == "__main__":
#     file_path = os.path.dirname(__file__)
#     if file_path != "":
#         os.chdir(file_path)

#     grid_path = "../config/input_grid.txt"
#     rules_path = "../config/grid_rules.config"
#     myGridworld = Gridworld(grid_path, rules_path)
#     myGridworld.print_grid()
