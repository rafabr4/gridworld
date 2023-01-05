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
    n_episodes = 100
    myDynaAgent = DynaAgent(
        myGridworld,
        # exploration="epsilon",
        # exploration="decaying-epsilon",
        # decay_eps_episodes=n_episodes,
        # epsilon=0.01,
        exploration="ucb",
        ucb_c=0.5,
        alfa=0.5,
        end_alfa=0.05,
        gamma=1,
    )

    steps_per_episode = [0] * n_episodes
    for i in range(n_episodes):
        myDynaAgent.init_round(method="default")
        while not myDynaAgent.finished():
            myDynaAgent.play_step(n_updates=50)
        steps_per_episode[i] = myDynaAgent.steps

    # Repeat with epsilon set to 0 to see how close are we to the optimal policy
    # myDynaAgent.epsilon = 0
    myDynaAgent.ucb_c = 0
    myDynaAgent.init_round(method="default")
    while not myDynaAgent.finished():
        myDynaAgent.play_step(n_updates=0, verbose=True)
    final_steps = myDynaAgent.steps
    pass
