# import des packages

import numpy as np
import gym
import matplotlib
import time

env = gym.make("BipedalWalker-v3")

def Random_games():
    # Each of this episode is its own game.
    action_size = env.action_space.shape[0]

    for episode in range(5):
        env.reset()
        timer_init = time.time()
        # this is each frame, up to 500...but we wont make it that far with random.
        while True:
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be any of one how in list on 4, for example [0 1 0 0]
            action = np.random.uniform(-1.0, 1.0, size = action_size)

            timer = time.time() - timer_init

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            next_state, reward, done, info = env.step(action)

            print("{:.2f}".format(timer))
            # lets print everything in one line:
            #print(reward, action)
            if done or timer > 5:
                break

    env.close()

Random_games()
