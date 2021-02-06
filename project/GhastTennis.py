from __future__ import print_function
from builtins import range
import MalmoPython
import os
import sys
import time
import json
import functools
print = functools.partial(print, flush=True)

class Agent:
    def __init__(self, host):
        self.host = host

    def start(self, mission, mission_record):
        # Start mission
        try:
            self.host.startMission(mission, mission_record)
        except RuntimeError as e:
            print('Error starting mission', e)
            exit(1)

        # Wait for mission to start
        print("Waiting for the mission to start ", end=' ')
        world_state = self.host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = self.host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)

        # Initialize mission
        print()
        print("Mission running ", end=' ')
        self.initialize()

        # Take action
        while world_state.is_mission_running:
            print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            self.takeAction(world_state)
            for error in world_state.errors:
                print("Error:",error.text)

        # End mission
        print()
        print("Mission ended")

    def initialize(self):
        self.cleanWorld()
        self.makeInvincible()
        time.sleep(0.1)
        self.summonGhast(0, 10, 0)

    def takeAction(self, world_state):
        ghasts, fireballs = self.getGhastsAndFireballs(world_state)
        print('Ghasts:', ghasts)
        print('Fireballs:', fireballs)

    def cleanWorld(self):
        self.host.sendCommand('chat /entitydata @e[type=Ghast] {DeathLootTable:"minecraft:empty"}')
        time.sleep(0.1)
        self.host.sendCommand('chat /kill @e[type=!Player]')

    def makeInvincible(self):
        self.host.sendCommand('chat /effect @p 11 10000 255 True')

    def summonGhast(self, x, y, z, stationary=True):
        if stationary:
            self.host.sendCommand(f'chat /summon minecart {x} {y} {z} {{NoGravity:1, Passengers:[{{id:Ghast}}]}}')
        else:
            self.host.sendCommand(f'chat /summon Ghast {x} {y} {z}')

    def getGhastsAndFireballs(self, world_state):
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
        return ghasts, fireballs

if __name__ == '__main__':
    # Create default Malmo objects
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    # Setup mission
    mission_file = './mission.xml'
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        mission = MalmoPython.MissionSpec(mission_xml, True)
    mission_record = MalmoPython.MissionRecordSpec()

    # Create agent
    agent = Agent(agent_host)
    agent.start(mission, mission_record)
