import os
from grid_env import Gridworld
from dyna_agent import DynaAgent

if __name__ == "__main__":
    file_path = os.path.dirname(__file__)
    if file_path != "":
        os.chdir(file_path)

    grid_path = "../config/input_grid.txt"
    rules_path = "../config/grid_rules.config"
    myGridworld = Gridworld(grid_path, rules_path)
    # myGridworld.print_grid()

    # TODO test random epsilon and alfa values, store results, plot and find best
    myDynaAgent = DynaAgent(myGridworld, epsilon=0.01, alfa=0.1, gamma=1)

    n_epsiodes = 100
    steps_per_episode = [0] * n_epsiodes
    for i in range(n_epsiodes):
        myDynaAgent.initialize_env(method="default")
        while not myDynaAgent.finished():
            myDynaAgent.play_e_greedy_step(n_updates=10)
            steps_per_episode[i] += 1

    # Repeat with epsilon set to 0 to see how close are we to the optimal policy
    myDynaAgent.epsilon = 0
    myDynaAgent.initialize_env(method="default")
    final_steps = 0
    while not myDynaAgent.finished():
        myDynaAgent.play_e_greedy_step(n_updates=0)
        final_steps += 1
    pass
