import unittest

from src import grid_env


grid_good_path = "test/config/input_grid_good.txt"
rules_good_path = "test/config/grid_rules_good.config"

grid_bad_01_path = "test/config/input_grid_bad_01.txt"  # Invalid character
grid_bad_02_path = "test/config/input_grid_bad_02.txt"  # Invalid character
grid_bad_03_path = "test/config/input_grid_bad_03.txt"  # Different row lengths
grid_bad_04_path = "test/config/input_grid_bad_04.txt"  # Different row lengths
grid_bad_05_path = "test/config/input_grid_bad_05.txt"  # Row with no elements
grid_bad_06_path = "test/config/input_grid_bad_06.txt"  # Invalid line
grid_bad_07_path = "test/config/input_grid_bad_07.txt"  # More than one 'S'
grid_bad_08_path = "test/config/input_grid_bad_08.txt"  # More than one 'G'

rules_bad_01_path = "test/config/grid_rules_bad_01.config"  # Invalid action


class TestInputGridLoadingPositive(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)

    def test_dot_space_01(self):
        self.assertEqual(self.gridworld.grid[0][0], ".", "Incorrect grid loading for '.'")

    def test_dot_space_02(self):
        self.assertEqual(self.gridworld.grid[3][3], ".", "Incorrect grid loading for '.'")

    def test_X_space_01(self):
        self.assertEqual(self.gridworld.grid[1][3], "X", "Incorrect grid loading for 'X'")

    def test_X_space_02(self):
        self.assertEqual(self.gridworld.grid[3][2], "X", "Incorrect grid loading for 'X'")

    def test_S_space(self):
        self.assertEqual(self.gridworld.grid[3][0], "S", "Incorrect grid loading for 'S'")

    def test_G_space(self):
        self.assertEqual(self.gridworld.grid[2][4], "G", "Incorrect grid loading for 'G'")


class TestInputGridLoadingNegative(unittest.TestCase):
    def test_invalid_char_01(self):
        with self.assertRaises(grid_env.InvalidGridError):
            grid_env.Gridworld(grid_bad_01_path, rules_good_path)

    def test_invalid_char_02(self):
        with self.assertRaises(grid_env.InvalidGridError):
            grid_env.Gridworld(grid_bad_02_path, rules_good_path)

    def test_different_row_len_01(self):
        with self.assertRaises(grid_env.InvalidGridError):
            grid_env.Gridworld(grid_bad_03_path, rules_good_path)

    def test_different_row_len_02(self):
        with self.assertRaises(grid_env.InvalidGridError):
            grid_env.Gridworld(grid_bad_04_path, rules_good_path)

    def test_row_no_elements(self):
        with self.assertRaises(grid_env.InvalidGridError):
            grid_env.Gridworld(grid_bad_05_path, rules_good_path)

    def test_invalid_line(self):
        with self.assertRaises(grid_env.InvalidGridError):
            grid_env.Gridworld(grid_bad_06_path, rules_good_path)

    def test_multiple_S(self):
        with self.assertRaises(grid_env.InvalidGridError):
            grid_env.Gridworld(grid_bad_07_path, rules_good_path)

    def test_multiple_G(self):
        with self.assertRaises(grid_env.InvalidGridError):
            grid_env.Gridworld(grid_bad_08_path, rules_good_path)


class TestConfigLoadingPositive(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)

    def test_actions_loading(self):
        self.assertEqual(
            self.gridworld.actions, set(["L", "R", "U", "D"]), "Incorrect actions loading"
        )

    def test_default_reward_loading(self):
        self.assertEqual(self.gridworld.default_reward, -1, "Incorrect default reward loading")

    def test_custom_reward_loading(self):
        self.assertEqual(
            self.gridworld.custom_rewards["1,4-D"], 0, "Incorrect custom reward loading"
        )
        self.assertEqual(
            self.gridworld.custom_rewards["2,3-R"], 0, "Incorrect custom reward loading"
        )
        self.assertEqual(
            self.gridworld.custom_rewards["3,4-U"], 0, "Incorrect custom reward loading"
        )


class TestConfigLoadingNegative(unittest.TestCase):
    def test_invalid_action(self):
        with self.assertRaises(grid_env.InvalidActionError):
            grid_env.Gridworld(grid_good_path, rules_bad_01_path)


# TODO missing testing for:
#  - initialize
#    - default
#    - random
#    - state
#    - state (negative, could fail in various ways: off grid, non int, etc.)
#  - get_possible_actions
#  - take_action
