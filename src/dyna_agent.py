from base_env import Environment
import numpy as np
import random


class DynaAgent:
    def __init__(self, env: Environment, epsilon: float = 0.1, alfa: float = 0.1, gamma: float = 1):
        self.env = env
        self.epsilon = epsilon
        self.alfa = alfa
        self.gamma = gamma
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
        # TODO implement a non-deterministic model
        # Keep track of last X observed new_states for each state_action pair,
        # and define a probability distribution for transitioning to each one.
        # Then in _do_planning sample the new state according to the distribution.
        self._model[state_action] = (reward, new_state)

    def _do_planning(self, n_updates) -> None:
        for _ in range(n_updates):
            state = random.choice(list(self._seen_states.keys()))
            action = random.choice(list(self._seen_states[state]))
            reward, new_state = self._model[(state, action)]
            self._update_qvalue(state, action, reward, new_state)

    def initialize_env(self, method: str = None, state=None) -> None:
        # We're guaranteed to initialize in a non-terminal state
        if method in ["default", "random"]:
            self.current_state, self.current_cell = self.env.initialize(method=method)
        elif state is not None:
            self.current_state, self.current_cell = self.env.initialize(state=state)
        else:
            # TODO raise exception
            pass

    def play_e_greedy_step(self, n_updates: int = 10) -> None:
        actions = self.env.get_possible_actions(self.current_state)

        # TODO implement decaying epsilon
        # TODO implement r + c*sqrt(f(t)) exploration bonus strategy for action selection

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

        # Update seen states and actions taken
        if self.current_state in self._seen_states:
            self._seen_states[self.current_state].update(action)
        else:
            self._seen_states[self.current_state] = set([action])

        reward, new_state = self.env.take_action(action)

        self._update_qvalue(self.current_state, action, reward, new_state)
        self._update_model((self.current_state, action), reward, new_state)
        self.current_state = self.env.current_state
        self.current_cell = self.env.current_cell

        self._do_planning(n_updates)

    def finished(self) -> bool:
        return self.current_cell == "G"
