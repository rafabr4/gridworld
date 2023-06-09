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
    def test_get_possible_actions_current_dot_01(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(0, 0))
        actions = self.gridworld.get_possible_actions()
        self.assertEqual({"L", "R", "U", "D"}, set(actions))

    def test_get_possible_actions_current_dot_02(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(2, 3))
        actions = self.gridworld.get_possible_actions()
        self.assertEqual({"L", "R", "U", "D"}, set(actions))

    def test_get_possible_actions_current_S(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(3, 0))
        actions = self.gridworld.get_possible_actions()
        self.assertEqual({"L", "R", "U", "D"}, set(actions))

    def test_get_possible_actions_current_X(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        # Access internal variable to force state
        self.gridworld._current_state = (2, 2)
        actions = self.gridworld.get_possible_actions()
        self.assertEqual(set(), set(actions))

    def test_get_possible_actions_current_G(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        # Access internal variable to force state
        self.gridworld._current_state = (2, 4)
        actions = self.gridworld.get_possible_actions()
        self.assertEqual(set(), set(actions))

    def test_get_possible_actions_specific_dot_01(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        actions = self.gridworld.get_possible_actions((0, 0))
        self.assertEqual({"L", "R", "U", "D"}, set(actions))

    def test_get_possible_actions_specific_dot_02(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        actions = self.gridworld.get_possible_actions((2, 3))
        self.assertEqual({"L", "R", "U", "D"}, set(actions))

    def test_get_possible_actions_specific_S(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        actions = self.gridworld.get_possible_actions((3, 0))
        self.assertEqual({"L", "R", "U", "D"}, set(actions))

    def test_get_possible_actions_specific_X(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        actions = self.gridworld.get_possible_actions((2, 2))
        self.assertEqual(set(), set(actions))

    def test_get_possible_actions_specific_G(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        actions = self.gridworld.get_possible_actions((2, 4))
        self.assertEqual(set(), set(actions))

    def test_get_all_possible_states(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        states = self.gridworld.get_all_possible_states()
        self.assertEqual(
            states,
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 4),
                (2, 0),
                (2, 1),
                (2, 3),
                (2, 4),
                (3, 0),
                (3, 1),
                (3, 3),
                (3, 4),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 4),
                (5, 0),
                (5, 1),
                (5, 2),
                (5, 3),
                (5, 4),
            ],
        )

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

    def test_state_dynamics_dot_1_2(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(1, 2))
        self.assertEqual(
            self.gridworld.transitions[1][2]["L"], (1, 1, -1), "Incorrect dynamic at 1,2"
        )
        self.assertEqual(
            self.gridworld.transitions[1][2]["R"], (1, 2, -1), "Incorrect dynamic at 1,2"
        )
        self.assertEqual(
            self.gridworld.transitions[1][2]["U"], (0, 2, -1), "Incorrect dynamic at 1,2"
        )
        self.assertEqual(
            self.gridworld.transitions[1][2]["D"], (1, 2, -1), "Incorrect dynamic at 1,2"
        )

    def test_state_dynamics_dot_2_3(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(2, 3))
        self.assertEqual(
            self.gridworld.transitions[2][3]["L"], (2, 3, -1), "Incorrect dynamic at 2,3"
        )
        self.assertEqual(
            self.gridworld.transitions[2][3]["R"], (2, 4, 0), "Incorrect dynamic at 2,3"
        )
        self.assertEqual(
            self.gridworld.transitions[2][3]["U"], (2, 3, -1), "Incorrect dynamic at 2,3"
        )
        self.assertEqual(
            self.gridworld.transitions[2][3]["D"], (3, 3, -1), "Incorrect dynamic at 2,3"
        )

    def test_state_dynamics_S_3_0(self):
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

    def test_take_action_01(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(0, 0))
        reward, new_state = self.gridworld.take_action("L")
        self.assertEqual(reward, -1)
        self.assertEqual(new_state, (0, 0))

    def test_take_action_02(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(3, 0))
        reward, new_state = self.gridworld.take_action("R")
        self.assertEqual(reward, -1)
        self.assertEqual(new_state, (3, 1))

    def test_take_action_03(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(3, 3))
        reward, new_state = self.gridworld.take_action("D")
        self.assertEqual(reward, -1)
        self.assertEqual(new_state, (3, 3))

    def test_take_action_04(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(3, 4))
        reward, new_state = self.gridworld.take_action("U")
        self.assertEqual(reward, 0)
        self.assertEqual(new_state, (2, 4))


class TestDynamicsNegative(unittest.TestCase):
    def test_get_possible_actions_specific_off_grid_01(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        with self.assertRaises(grid_env.InvalidStateError):
            self.gridworld.get_possible_actions((0, -1))

    def test_get_possible_actions_specific_off_grid_02(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        with self.assertRaises(grid_env.InvalidStateError):
            self.gridworld.get_possible_actions((7, 1))

    def test_take_invalid_action_from_dot(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(0, 0))
        with self.assertRaises(grid_env.InvalidActionError):
            self.gridworld.take_action("P")

    def test_take_invalid_action_from_S(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        self.gridworld.initialize(state=(3, 0))
        with self.assertRaises(grid_env.InvalidActionError):
            self.gridworld.take_action("T")

    def test_take_action_from_X(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        # Force to be in an 'X' cell
        self.gridworld._current_state = (1, 3)
        self.gridworld._current_cell = "X"
        with self.assertRaises(grid_env.InvalidActionError):
            self.gridworld.take_action("L")

    def test_take_action_from_G(self):
        self.gridworld = grid_env.Gridworld(grid_good_path, rules_good_path)
        # Force to be in the 'G' cell
        self.gridworld._current_state = (2, 4)
        self.gridworld._current_cell = "G"
        with self.assertRaises(grid_env.InvalidActionError):
            self.gridworld.take_action("U")


# TODO add tests for custom transitions
