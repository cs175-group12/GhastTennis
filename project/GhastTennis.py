from __future__ import print_function
from builtins import range
import MalmoPython
import os
import sys
import time
import json
import numpy as np
import random
import functools
print = functools.partial(print, flush=True)

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo

class Agent(gym.Env):
    def __init__(self, env_config):
        # Static parameters
        self.obs_size = 3
        self.max_episode_steps = 100
        # TODO add continuous actions for yaw and pitch
        self.action_dict = {
            0: 'attack 0',
            1: 'attack 1'  
        }

        # Rllib parameters
        # TODO make action space continuous
        self.action_space = Discrete(len(self.action_dict))
        self.observation_space = Box(-100, 100, shape=(2 * self.obs_size * self.obs_size, ), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # Agent parameters
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []
        self.num_ghasts = 0
        self.num_fireballs = 0
        self.life = 20
        self.damage_taken = 0

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0
        self.num_ghasts = 0
        self.num_fireballs = 0
        self.life = 20
        self.damage_taken = 0

        # Get Observation
        self.obs = self.get_observation(world_state)

        return self.obs

    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        # Setup mission
        mission_file = './mission.xml'
        with open(mission_file, 'r') as f:
            print("Loading mission from %s" % mission_file)
            mission_xml = f.read()
            mission = MalmoPython.MissionSpec(mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()

        my_mission = MalmoPython.MissionSpec(mission_xml, True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'Agent' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)
        self.initialize()
        return world_state

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
        # Get Action
        command = self.action_dict[action]
        self.agent_host.sendCommand(command)
        time.sleep(0.2)
        self.episode_step += 1  
        # print(self.episode_step)

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state) 

        # Get Done
        done = not world_state.is_mission_running 
        
        # Get Reward
        reward = 0
        if self.num_ghasts == 0: # killed all ghasts
            print("killed ghasts")
            self.agent_host.sendCommand('quit')
            reward += 15 

        if self.obs[8] < 0: # fireball redirect
            reward += 1
        reward -= self.damage_taken
        # for r in world_state.rewards:
        #     reward += r.getValue()

        print("reward: {}".format(reward))
        self.episode_return += reward
        print("episode_return {}".format(self.episode_return))

        return self.obs, reward, done, dict()

    def get_observation(self, world_state):
        # to simplify problem, only feeding x y z position of ghast fireball as observations
        """
        Use the agent observation API to get a flattened 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> the state observation
        """     
        ghasts, fireballs = self.getGhastsAndFireballs(world_state)
        obs = np.zeros((2 * self.obs_size * self.obs_size, ))
            
        # TODO edit to work with multiple ghasts
        if (len(ghasts) != 0):
            obs[0] = ghasts[0]["x"]
            obs[1] = ghasts[0]["y"]
            obs[2] = ghasts[0]["z"]
        if (len(fireballs) != 0):
            # print("motionX: {}, motionY: {}, motionZ: {}".format(fireballs[0]["motionX"], fireballs[0]["motionY"], fireballs[0]["motionZ"]))
            obs[3] = fireballs[0]["x"]
            obs[4] = fireballs[0]["y"]
            obs[5] = fireballs[0]["z"]
            obs[6] = fireballs[0]["motionX"]
            obs[7] = fireballs[0]["motionY"]
            obs[8] = fireballs[0]["motionZ"]

        print(obs)

        return obs

        # ------------------------------------------------------------------------------------

    def initialize(self):
        print("initializing")
        self.cleanWorld()
        # self.makeInvincible()
        time.sleep(0.1)
        self.summonGhast(random.randint(-10, 10), 3, -20)
        time.sleep(1)

    def cleanWorld(self):
        self.agent_host.sendCommand('chat /entitydata @e[type=Ghast] {DeathLootTable:"minecraft:empty"}')
        time.sleep(0.1)
        self.agent_host.sendCommand('chat /kill @e[type=!Player]')

    def makeInvincible(self):
        self.agent_host.sendCommand('chat /effect @p 11 10000 255 True')

    def summonGhast(self, x, y, z, yaw=0, stationary=True):
        if stationary:
            self.agent_host.sendCommand(f'chat /summon minecart {x} {y} {z} {{NoGravity:1, Passengers:[{{id:Ghast, Rotation:[{yaw}f, 0f]}}]}}')
        else:
            self.agent_host.sendCommand(f'chat /summon Ghast {x} {y} {z} {{Rotation:[{yaw}f, 0f]}}')

    def getGhastsAndFireballs(self, world_state):
        ''' checks world state for fireballs and ghasts and adds them into seperate lists '''
        if world_state.number_of_observations_since_last_state == 0:
            return [], []
        obvsText = world_state.observations[-1].text
        data = json.loads(obvsText)
        if 'entities' not in data:
            return [], []
        ghasts = []
        fireballs = []
        for entity in data['entities']:
            if entity['name'] == 'Ghast':
                ghasts.append(entity)
            elif entity['name'] == 'Fireball':
                fireballs.append(entity)
            elif entity['name'] == "GhastTennis Agent":
                # check if agent took damage (for negative rewards)
                life = entity['life']
                print("Life: {}".format(life))
                print("self life {}".format(self.life))
                if life < self.life:
                    self.damage_taken = self.life - life
                    print("Damage Taken: {}".format(self.damage_taken))
                    self.life = entity['life']
                else:
                    self.life = life
                    self.damage_taken = 0

        # Keep track of number of ghasts and fireballs 
        self.num_ghasts = len(ghasts)
        self.num_fireballs = len(fireballs)
        return ghasts, fireballs

if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=Agent, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())
