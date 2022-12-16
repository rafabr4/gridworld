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
grid_bad_09_path = "test/config/input_grid_bad_09.txt"  # No 'S'
grid_bad_10_path = "test/config/input_grid_bad_10.txt"  # No 'G'

rules_bad_01_path = "test/config/grid_rules_bad_01.config"  # Invalid action definition
rules_bad_02_path = "test/config/grid_rules_bad_02.config"  # Invalid default reward format
rules_bad_03_path = "test/config/grid_rules_bad_03.config"  # Invalid custom reward format
rules_bad_05_path = "test/config/grid_rules_bad_04.config"  # Invalid custom reward action
rules_bad_04_path = "test/config/grid_rules_bad_05.config"  # Invalid custom reward off grid state


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

    def test_default_start(self):
        self.assertEqual(
            self.gridworld._default_start_state, (3, 0), "Incorrect default start state"
        )

    def test_goal_state(self):
        self.assertEqual(self.gridworld._goal_state, (2, 4), "Incorrect goal state")


class TestInputGridLoadingNegative(unittest.TestCase):
    def test_invalid_char_01(self):
        # Grid with invalid character 'T'
        with self.assertRaises(grid_env.InvalidGridError):
            grid_env.Gridworld(grid_bad_01_path, rules_good_path)

    def test_invalid_char_02(self):
        # Grid with invalid character ' '
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

    def test_no_S(self):
        with self.assertRaises(grid_env.InvalidGridError):
            grid_env.Gridworld(grid_bad_09_path, rules_good_path)

    def test_no_G(self):
        with self.assertRaises(grid_env.InvalidGridError):
            grid_env.Gridworld(grid_bad_10_path, rules_good_path)


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
    def test_invalid_action_definition(self):
        # Defining action 'P' is not valid
        with self.assertRaises(grid_env.InvalidActionError):
            grid_env.Gridworld(grid_good_path, rules_bad_01_path)

    def test_invalid_default_reward_format(self):
        # Using ':' instead of '='
        with self.assertRaises(grid_env.InvalidRewardConfigError):
            grid_env.Gridworld(grid_good_path, rules_bad_02_path)

    def test_invalid_custom_reward_format(self):
        # Using ',' instead of '-'
        with self.assertRaises(grid_env.InvalidRewardConfigError):
            grid_env.Gridworld(grid_good_path, rules_bad_03_path)

    def test_invalid_custom_reward_action(self):
        # Invalid action 'P' in custom reward
        with self.assertRaises(grid_env.InvalidRewardConfigError):
            grid_env.Gridworld(grid_good_path, rules_bad_04_path)

    def test_invalid_custom_reward_state(self):
        # Invalid off grid state (8, 3)
        with self.assertRaises(grid_env.InvalidRewardConfigError):
            grid_env.Gridworld(grid_good_path, rules_bad_05_path)


class TestInitializePositive(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)

    def test_initialize_default(self):
        self.gridworld.initialize(method="default")
        self.assertEqual(self.gridworld.current_state, (3, 0), "Incorrect default initialization")

    def test_initialize_random(self):
        for _ in range(50):
            self.gridworld.initialize(method="random")
            # Row value should be within grid
            self.assertGreaterEqual(
                self.gridworld.current_state[0], 0, "Incorrect random initialization"
            )
            self.assertLess(self.gridworld.current_state[0], 6, "Incorrect random initialization")

            # Column value should be within grid
            self.assertGreaterEqual(
                self.gridworld.current_state[1], 0, "Incorrect random initialization"
            )
            self.assertLess(self.gridworld.current_state[1], 5, "Incorrect random initialization")

            # Value should be '.' or 'S'
            self.assertIn(
                self.gridworld.current_cell, [".", "S"], "Incorrect random initialization"
            )

    def test_initialize_specific_state_01(self):
        # Initialize to a '.'
        self.gridworld.initialize(state=(0, 0))
        self.assertEqual(self.gridworld.current_state, (0, 0), "Incorrect specific initialization")
        self.assertEqual(self.gridworld.current_cell, ".", "Incorrect specific initialization")

    def test_initialize_specific_state_02(self):
        # Initialize to an 'S'
        self.gridworld.initialize(state=(3, 0))
        self.assertEqual(self.gridworld.current_state, (3, 0), "Incorrect specific initialization")
        self.assertEqual(self.gridworld.current_cell, "S", "Incorrect specific initialization")


class TestInitializeNegative(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)

    def test_initialize_invalid_state_01(self):
        # Pass a list instead of a tuple
        with self.assertRaises(grid_env.InvalidStateError):
            self.gridworld.initialize(state=[0, 0])

    def test_initialize_invalid_state_02(self):
        # Pass a string instead of a tuple
        with self.assertRaises(grid_env.InvalidStateError):
            self.gridworld.initialize(state="state")

    def test_initialize_invalid_state_03(self):
        # Tuple larger than expected
        with self.assertRaises(grid_env.InvalidStateError):
            self.gridworld.initialize(state=(0, 0, 0))

    def test_initialize_invalid_state_04(self):
        # Tuple shorter than expected
        with self.assertRaises(grid_env.InvalidStateError):
            self.gridworld.initialize(state=(0))

    def test_initialize_invalid_state_05(self):
        # Tuple of strings instead of ints
        with self.assertRaises(grid_env.InvalidStateError):
            self.gridworld.initialize(state=("a", "b"))

    def test_initialize_invalid_state_06(self):
        # Off grid state
        with self.assertRaises(grid_env.InvalidStateError):
            self.gridworld.initialize(state=(-1, 0))

    def test_initialize_invalid_state_07(self):
        # Off grid state
        with self.assertRaises(grid_env.InvalidStateError):
            self.gridworld.initialize(state=(0, 7))

    def test_initialize_invalid_state_08(self):
        # Try to initialize to an 'X'
        with self.assertRaises(grid_env.InvalidStateError):
            self.gridworld.initialize(state=(1, 3))

    def test_initialize_invalid_state_09(self):
        # Try to initialize to a 'G'
        with self.assertRaises(grid_env.InvalidStateError):
            self.gridworld.initialize(state=(2, 4))


class TestChangeStateNegative(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)

    def test_change_state(self):
        with self.assertRaises(grid_env.IlegalStateChangeError):
            self.gridworld.current_state = (0, 0)

    def test_change_cell(self):
        with self.assertRaises(grid_env.IlegalCellChangeError):
            self.gridworld.current_cell = "."


class TestDynamicsPositive(unittest.TestCase):
    def test_get_possible_actions_dot_01(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(0, 0))
        actions = self.gridworld.get_possible_actions()
        self.assertEqual({"L", "R", "U", "D"}, set(actions))

    def test_get_possible_actions_dot_02(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(2, 3))
        actions = self.gridworld.get_possible_actions()
        self.assertEqual({"L", "R", "U", "D"}, set(actions))

    def test_get_possible_actions_S(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(3, 0))
        actions = self.gridworld.get_possible_actions()
        self.assertEqual({"L", "R", "U", "D"}, set(actions))

    def test_get_possible_actions_X(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld._current_state = (2, 2)
        actions = self.gridworld.get_possible_actions()
        self.assertEqual(set(), set(actions))

    def test_get_possible_actions_G(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld._current_state = (2, 4)
        actions = self.gridworld.get_possible_actions()
        self.assertEqual(set(), set(actions))

    def test_state_dynamics_dot_0_0(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(0, 0))
        self.assertEqual(
            self.gridworld.transitions[0][0]["L"], (0, 0, -1), "Incorrect dynamic at 0,0"
        )
        self.assertEqual(
            self.gridworld.transitions[0][0]["R"], (0, 1, -1), "Incorrect dynamic at 0,0"
        )
        self.assertEqual(
            self.gridworld.transitions[0][0]["U"], (0, 0, -1), "Incorrect dynamic at 0,0"
        )
        self.assertEqual(
            self.gridworld.transitions[0][0]["D"], (1, 0, -1), "Incorrect dynamic at 0,0"
        )

    def test_state_dynamics_dot_3_0(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(3, 0))
        self.assertEqual(
            self.gridworld.transitions[3][0]["L"], (3, 0, -1), "Incorrect dynamic at 3,0"
        )
        self.assertEqual(
            self.gridworld.transitions[3][0]["R"], (3, 1, -1), "Incorrect dynamic at 3,0"
        )
        self.assertEqual(
            self.gridworld.transitions[3][0]["U"], (2, 0, -1), "Incorrect dynamic at 3,0"
        )
        self.assertEqual(
            self.gridworld.transitions[3][0]["D"], (4, 0, -1), "Incorrect dynamic at 3,0"
        )

    # TODO continue with dynamics with 1,2, 2,3


# TODO missing testing for:
#  - get_possible_actions
#  - take_action (positive: returns expected new state and reward)
#  - take_action (negative: since we can't be in 'X' or 'G', pass something like 'P'?,
#                           or force 'G' and pass an action 'D' and see it fail?)
