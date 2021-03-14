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
    # layersizes = np.random.randint(low = 9, high = 81, size = np.random.random_integers(3,6))
    # layersizes[0] = 9   #inputsize
    # layersizes[-1] = 5  #outputsize
    trainedAI = NetworkV3([1])

    trainedAI.loadtxt(0) # load trained agent from files

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
        self.virtualWorld.player.set_AI(self.trainedAI.predict)


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
        player, ghasts, fireballs = self.getObservations(world_state)
        # print(f"Ghast: {ghasts}")
        # print(f"Fireballs: {fireballs}")

        if(len(ghasts) == 0 or len(fireballs) == 0): #if either are missing do nothing
            return

        # get player data
        playerPos = np.array([player['x'], player['y'], player['z']])
        pitch = player['pitch']
        yaw = player['yaw']

        # get closest ghast and fireball
        ghast = self.getClosestEntity(playerPos, ghasts)
        fireball = self.getClosestEntity(playerPos, fireballs)

		# get positions
        ghastPos = np.array([ghast['x'], ghast['y'], ghast['z']])
        fireballPos = np.array([fireball['x'], fireball['y'], fireball['z']])
        fireballVelocity = np.array([fireball['motionX'], fireball['motionY'], fireball['motionZ']])

        # set player position and rotation here
        self.virtualWorld.player.transform.position = playerPos
        self.virtualWorld.player.transform.set_rotation(pitch,yaw) 

        # create observationData
        ghastPoint = self.virtualWorld.player.transform.world_to_local(ghastPos)
        fireballPoint = self.virtualWorld.player.transform.world_to_local(fireballPos)
        fireballVel = self.virtualWorld.player.transform.world_to_local(fireballVelocity,direction=True)
        observationData = np.asarray([ghastPoint,fireballPoint,fireballVel]).reshape(9,1)

        # get cmd from brain function
        cmd = self.virtualWorld.player.brain(observationData)

        move = f"move {cmd[0][0] * 2 -1 }"
        strafe = f"strafe {cmd[1][0] * 2 - 1}"
        pitch = f"pitch {cmd[2][0] * 2 - 1}"
        turn = f"turn {cmd[3][0] * 2 - 1}"
        attack = f"attack {1 if cmd[4][0] > 0 else 0}"

        # run cmds
        self.agent_host.sendCommand(move)
        #time.sleep(0.1)
        self.agent_host.sendCommand(strafe)
        #time.sleep(0.1)
        self.agent_host.sendCommand(pitch)
        #time.sleep(0.1)
        self.agent_host.sendCommand(turn)
        #time.sleep(0.1)
        self.agent_host.sendCommand(attack)
        time.sleep(0.05)

    def getObservations(self, world_state):
        '''
        Gets the player, ghasts, and fireballs from the world state.
        '''

        if world_state.number_of_observations_since_last_state == 0:
            return None, [], []
        obvsText = world_state.observations[-1].text
        data = json.loads(obvsText)
        if 'entities' not in data:
            return None, [], []
        player = None
        ghasts = []
        fireballs = []
        for entity in data['entities']:
            if entity['name'] == 'Ghast':
                ghasts.append(entity)
            elif entity['name'] == 'Fireball':
                fireballs.append(entity)
            elif entity['name'] == 'GhastTennisAgent':
                player = entity
        return player, ghasts, fireballs

    def getClosestEntity(self, playerPos, entities):
        '''
        Get the closest entity relative to the agent.
        '''

        playerX = playerPos[0]
        playerY = playerPos[1]
        playerZ = playerPos[2]
        
        closest = None
        closestDistance = -1
        for entity in entities:
            entityX = entity['x']
            entityY = entity['y']
            entityZ = entity['z']

            distance = math.sqrt((playerX - entityX) ** 2 + (playerY - entityY) ** 2 + (playerZ - entityZ) ** 2)

            if closest is None or distance < closestDistance:
                closest = entity
                closestDistance = distance
        return closest

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
