from neuralnetdebug import NetworkV3
import notminecraft
import numpy as np
import MalmoPython
import sys
import time
import json
import matplotlib.pyplot as plt
import random
import math

def main():
    #initialize random gen 1
    layersizes = np.random.randint(low = 9, high = 81, size = np.random.random_integers(3,6))
    layersizes[0] = 9   #inputsize
    layersizes[-1] = 5  #outputsize
    trainedAI = NetworkV3(layersizes)

    trainedAI.loadtxt(7) # load trained agent from files

    runs = 10

    # Create agent
    agent = Agent(trainedAI)
    for i in range(runs):
        agent.start()



class Agent():
    def __init__(self, trainedAI):
        self.trainedAI = trainedAI # store trained agent in Agent class
        self.virtualWorld = None

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

    def initialize(self):
        print("Initializing...")
        self.cleanWorld()
        self.makeInvincible()
        time.sleep(0.1)
        x = random.randint(-10, 10)
        self.summonGhast(x, 3, -20)

        # initialize virtual simulator
        # create virtual world
        self.virtualWorld = notminecraft.world()
        #self.virtualWorld.prepare_pickling()
        self.virtualWorld.player.set_AI(self.trainedAI.predict)
        # f = notminecraft.fireball(self.virtualWorld, xyz= (x, 3, -20))
        # self.virtualWorld.spawn(f)
        # g = notminecraft.ghast(self.virtualWorld, xyz= (x, 3, -20))
        # self.virtualWorld.spawn(g)
        # self.virtualWorld.start()

        time.sleep(1)

    def init_malmo(self):
        """
        Initialize new Malmo mission.
        """

        # Load the XML file and create mission spec & record.
        mission_file = './mission.xml'
        with open(mission_file, 'r') as f:
            print("Loading mission from %s" % mission_file)
            mission_xml = f.read()
            my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission_record = MalmoPython.MissionRecordSpec()
            my_mission.requestVideo(800, 500)
            my_mission.setViewpoint(1)

        # Attempt to start Malmo.
        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(my_mission, my_clients, my_mission_record, 0, 'Agent')
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        # Start the world.
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)
        self.initialize()
        return world_state

    def start(self):
        """
        Runs agent while world is still running
        """
        world_state = self.init_malmo()
        # Take action
        while world_state.is_mission_running:
            print(".", end="")
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            self.takeAction(world_state)
            for error in world_state.errors:
                print("Error:",error.text)

        # End mission
        print()
        print("Mission ended")

    def takeAction(self, world_state):
        # ghasts and fireballs from malmo
        ghasts, fireballs = self.getGhastsAndFireballs(world_state)
        # print(f"Ghast: {ghasts}")
        # print(f"Fireballs: {fireballs}")

        if(len(ghasts) == 0 or len(fireballs) == 0): #if either are missing do nothing
            return

        #**update to give closest ones**
        ghast = ghasts[0]
        fireball = fireballs[0]

        ghastPos = [ghast['x'], ghast['y'], ghast['z']]
        fireballPos = [fireball['x'], fireball['y'], fireball['z']]
        fireballVelocity = [fireball['motionX'], fireball['motionY'], fireball['motionZ']]

        # tranform to points
        ghastPoint = np.asarray(ghastPos)
        fireballPoint = np.asarray(fireballPos)
        fireballVel = np.asarray(fireballVelocity)

        #set player position and rotation here
        self.virtualWorld.player.transform.position = '''your positional information from malmo in a flattened numpy array'''
        self.virtualWorld.player.transform.set_rotation(pitch,yaw) '''get information from malmo'''


        # create observationData
        ghastPoint = self.virtualWorld.player.transform.world_to_local(ghastPoint)
        fireballPoint = self.virtualWorld.player.transform.world_to_local(fireballPoint)
        fireballVel = self.virtualWorld.player.transform.world_to_local(fireballVel,direction=True)
        observationData = np.asarray([ghastPoint,fireballPoint,fireballVel]).reshape(9,1)
        # observationData = np.array(observationData).reshape((9,1))
        # print(observationData)
        # self.virtualWorld.observationData = observationData

        # predict cmd from brain function
        cmd = self.virtualWorld.player.brain(observationData)

        # cmd = self.virtualWorld.player.cmd
        print(cmd)

    def getGhastsAndFireballs(self, world_state):
        '''
        Checks world state for fireballs and ghasts and adds them into seperate lists.
        '''

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
                # if entity['id'] not in self.fireballs:
                #     self.fireballs.append(entity['id']) # keep track of new fireballs

        return ghasts, fireballs

    def cleanWorld(self):
        '''
        Remove all entities that is not the agent.
        '''

        self.agent_host.sendCommand('chat /entitydata @e[type=Ghast] {DeathLootTable:"minecraft:empty"}')
        time.sleep(0.1)
        self.agent_host.sendCommand('chat /kill @e[type=!Player]')

    def makeInvincible(self):
        '''
        Make the agent invincible by using potion.
        '''

        self.agent_host.sendCommand('chat /effect @p 11 10000 255 True')

    def summonGhast(self, x, y, z, yaw=0, stationary=True):
        '''
        Summon a Ghast at specific coordinate.
        If stationary, then the summoned Ghast will be inside a minecart.
        '''

        if stationary:
            self.agent_host.sendCommand(f'chat /summon minecart {x} {y} {z} {{NoGravity:1, Passengers:[{{id:Ghast, Rotation:[{yaw}f, 0f]}}]}}')
        else:
            self.agent_host.sendCommand(f'chat /summon Ghast {x} {y} {z} {{Rotation:[{yaw}f, 0f]}}')

    # def reset(self):
    #     # Reset Malmo.  
    #     world_state = self.init_malmo()


if __name__ == '__main__':
    main()