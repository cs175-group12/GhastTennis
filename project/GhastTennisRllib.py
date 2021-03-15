import notminecraft
import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import gym, ray
from gym.spaces import Box
from ray.rllib.agents import ppo
import functools
from ray.tune.logger import pretty_print
print = functools.partial(print, flush=True)

class Agent(gym.Env):
    def __init__(self, env_config):
        # # Static parameters
        self.obs_size = 3
        self.max_episode_steps = 100
        self.log_frequency = 1

        # notminecraft parameters
        self.virtualWorld = notminecraft.world() # create virtual world

        # Rllib parameters
        self.action_space = Box(-1, 1, shape=(5, ), dtype=np.float32) # 0: move, 1: strafe, 2: pitch, 3: turn, 4: attack
        self.observation_space = Box(-5000.0, 5000.0, shape=(3 * self.obs_size * self.obs_size, ), dtype=np.float32) # 0: ghast position, 1: fireball position, 2: fireball velocity

        # # Agent parameters
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []


    def reset(self):
        """
        Resets the environment for the next episode.
        Returns
            observation: <np.array> flattened initial obseravtion
        """

        # Reset notminecraft world
        self.virtualWorld.reset()

        # Reset variables.
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0


        # Log last episode.
        if len(self.returns) > self.log_frequency + 1 and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get observation.
        # TODO: change obs to fit observation_space
        ghast, fireball = self.get_observation()
        if fireball == None:
            observationData = np.array([ghast, fireball, None])
        else:
            observationData = np.array([ghast, fireball, fireball.velocity])
        self.obs = observationData

        return self.obs

    def step(self, action):
        """
        Take an action in the environment and return the results.
        Args
            action: <int> index of the action to take
        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """
        # pitch/turn
        self.virtualWorld.player.turn(action[2], action[3])

        # attack
        if action[0] > 0:
            self.virtualWorld.player.attack()

        self.update_world()# update with new cmd
        self.update_rewards() 
        self.episode_step += 1

        # Get observation.
        ghast, fireball = self.get_observation()
        observationData = np.array([ghast, fireball, fireball.velocity])
        self.obs = observationData

        # Check reward.
        self.reward = self.virtualWorld.score

        return self.obs, reward

    def get_observation(self):
        """
        gets observationData from notminecraft world
        """
        return self.virtualWorld.observe()

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('GhastTennis Agent')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value))

if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=Agent, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    for i in range(1000):
        result = trainer.train()
        print(pretty_print(result))

        if (i % 2) == 0:
            checkpoint = trainer.save()
            print(checkpoint) # see checkpoint path
        i+=1