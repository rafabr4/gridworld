from base_env import Environment
import numpy as np
import random


class InvalidEnvInit(Exception):
    """
    Raised when trying to initialize the environment with a method that is not "default" or
    "random", and without specifying a state.
    """

    pass


class DynaAgent:
    def __init__(
        self,
        env: Environment,
        exploration: str = "epsilon",
        epsilon: float = 0.1,
        decay_eps_episodes: int = 100,
        alfa: float = 0.1,
        gamma: float = 1,
    ):
        self.env = env
        self.exploration = exploration
        if self.exploration in ["epsilon", "decaying-epsilon"]:
            self.start_epsilon = self.epsilon = epsilon
        if self.exploration in ["decaying-epsilon"]:
            self.decay_eps_episodes = decay_eps_episodes
        self.alfa = alfa
        self.gamma = gamma
        self.episodes = 0
        self.steps = 0
        self._build_state_action_pairs()
        self._initialize_model()
        self._initialize_qvalues()
        self._seen_states = {}

    def _build_state_action_pairs(self) -> None:
        self._states = self.env.get_all_possible_states()
        self._state_action_pairs = []
        for state in self._states:
            actions = self.env.get_possible_actions(state)
            # Terminal states won't be included since they have no actions
            for action in actions:
                self._state_action_pairs.append((state, action))
        self._state_action_pairs.sort()

    def _initialize_model(self) -> None:
        self._model = {}
        for state_action in self._state_action_pairs:
            self._model[state_action] = None

    def _initialize_qvalues(self) -> None:
        self._qvalues = {}
        for state in self._states:
            actions = self.env.get_possible_actions(state)
            if len(actions) > 0:
                self._qvalues[state] = {}
                for action in actions:
                    self._qvalues[state][action] = 0

    def _update_qvalue(self, state, action, reward, new_state) -> None:
        prev_qvalue = self._qvalues[state][action]
        if new_state in self._qvalues:
            max_value_new_state = max(self._qvalues[new_state].values())
        else:
            # We're in a (possibly terminal) state that has no actions
            max_value_new_state = 0
        td = reward + (self.gamma * max_value_new_state) - prev_qvalue
        self._qvalues[state][action] = prev_qvalue + self.alfa * td

    def _update_model(self, state_action, reward, new_state) -> None:
        # IMPROVEMENT implement a non-deterministic model
        # Keep track of last X observed new_states for each state_action pair,
        # and define a probability distribution for transitioning to each one.
        # Then in _do_planning sample the new state according to the distribution.
        # This would be useful for environments that are non-deterministic.
        self._model[state_action] = (reward, new_state)

    def _do_planning(self, n_updates) -> None:
        for _ in range(n_updates):
            state = random.choice(list(self._seen_states.keys()))
            action = random.choice(list(self._seen_states[state]))
            reward, new_state = self._model[(state, action)]
            self._update_qvalue(state, action, reward, new_state)

    def _update_epsilon(self):
        # Decay epsilon to 0 after decay_eps_episodes
        if (self.epsilon > 0) and (self.decay_eps_episodes >= self.episodes):
            r = (self.decay_eps_episodes - self.episodes) / self.decay_eps_episodes
            self.epsilon = r * self.epsilon

    def _get_e_greedy_action(self) -> str:
        actions = self.env.get_possible_actions(self.current_state)

        if np.random.uniform() <= self.epsilon:
            # Play randomly
            action = random.choice(actions)
        else:
            # Play greedy
            max_value = max(self._qvalues[self.current_state].values())
            action = random.choice(
                [
                    key
                    for key, value in self._qvalues[self.current_state].items()
                    if value == max_value
                ]
            )

        return action

    def _get_ucb_action(self) -> str:
        # TODO implement r + c*sqrt(f(t)) exploration bonus strategy for action selection
        return ""

    def _initialize_env(self, method: str = None, state=None) -> None:
        # We're guaranteed to initialize in a non-terminal state
        if method in ["default", "random"]:
            self.current_state, self.current_cell = self.env.initialize(method=method)
        elif state is not None:
            self.current_state, self.current_cell = self.env.initialize(state=state)
        else:
            raise InvalidEnvInit("Invalid environment initialization method!")

    def init_round(self, method: str = None, state=None) -> None:
        self.steps = 0
        self._initialize_env(method=method, state=state)

    def reset_agent(self) -> None:
        # TODO should just call __init__ to reset everything?
        self.episodes = 0
        if self.exploration in ["decaying-epsilon"]:
            self.epsilon = self.start_epsilon

    def play_step(self, n_updates: int = 10, verbose: bool = False) -> None:
        # Get an action according to the strategy being used
        if self.exploration in ["epsilon", "decaying-epsilon"]:
            action = self._get_e_greedy_action()
        else:
            action = self._get_ucb_action()

        # Update seen states and actions taken
        if self.current_state in self._seen_states:
            self._seen_states[self.current_state].update(action)
        else:
            self._seen_states[self.current_state] = set([action])

        if verbose:
            print(action)

        reward, new_state = self.env.take_action(action)

        self._update_qvalue(self.current_state, action, reward, new_state)
        self._update_model((self.current_state, action), reward, new_state)
        self.current_state = self.env.current_state
        self.current_cell = self.env.current_cell

        self._do_planning(n_updates)

        self.steps += 1

        if self.current_cell == "G":
            self.episodes += 1

            if self.exploration == "decaying-epsilon":
                self._update_epsilon()

            # TODO also update alfa (learning rate)?
            # Could use same strategy as for epsilon, but stablish a non-zero end value
            # See https://stackoverflow.com/questions/53198503/epsilon-and-learning-rate-decay-in-epsilon-greedy-q-learning

    def finished(self) -> bool:
        return self.current_cell == "G"


# IMPROVEMENT
# Define properties, getter, setter, etc.
