from __future__ import print_function
from builtins import range
import MalmoPython
import os
import sys
import time
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
        self.makeInvincible()
        self.summonGhast(0, 10, 0)

    def takeAction(self, world_state):
        pass

    def makeInvincible(self):
        self.host.sendCommand('chat /effect @p 11 10000 255 True')

    def summonGhast(self, x, y, z):
        self.host.sendCommand(f'chat /summon Ghast {x} {y} {z}')

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
