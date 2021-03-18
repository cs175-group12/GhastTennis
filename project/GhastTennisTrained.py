import sys
import time
import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import MalmoPython
import notminecraft
from neuralnetdebug import NetworkV3, PerfectNetwork,NeuralNetV4

def main():
    # Load NN data.


    trainedAI=NeuralNetV4.load(11)
    #trainedAI = PerfectNetwork()
    #trainedAI.loadtxt(120) # load trained agent from files
    # Create agent.
    runs = 12
    agent = Agent(trainedAI)
    for i in range(runs):
        agent.start(i + 1)



class Agent():
    def __init__(self, trainedAI):
        # Agent Parameters
        self.trainedAI = trainedAI
        self.virtualWorld = None
        self.playerPos = None
        self.playerPitch = 0
        self.playerYaw = 0

        # Malmo Host
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

    def start(self, run_iteration):
        """
        Start the agent.
        """

        # Initialize Malmo and the Minecraft world.
        print(f'RUN {run_iteration} STARTED')
        world_state = self.initMalmo()
        self.initWorld()

        # Initialize virtual world for the NN.
        self.virtualWorld = notminecraft.world()
        self.virtualWorld.player.set_AI(self.trainedAI.predict)
        time.sleep(1)

        # Take action while the mission is running.
        while world_state.is_mission_running:
            world_state = self.agent_host.getWorldState()
            self.takeAction(world_state)
            for error in world_state.errors:
                print('Error:', error.text)
                exit(1)

        # End mission.
        print(f'RUN {run_iteration} ENDED')
        print()

    def initMalmo(self):
        """
        Initialize new Malmo mission.
        """

        # Load the XML file and create mission spec & record.
        mission_file = './mission.xml'
        with open(mission_file, 'r') as f:
            print(f'Loading mission from {mission_file}')
            mission_xml = f.read()
            my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission_record = MalmoPython.MissionRecordSpec()

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
                    print('Error starting mission:', e)
                    exit(1)
                else:
                    time.sleep(2)

        # Start the world.
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print('Error:', error.text)
                exit(1)
        return world_state

    def initWorld(self):
        '''
        Initialize the Minecraft world.
        Clean the world, make the player invincible, and spawn ghasts.
        '''

        self.playerPos = np.array([0, 3, 0])
        self.playerYaw = 0
        self.playerPitch = 0

        self.cleanWorld()
        self.makeInvincible()
        time.sleep(0.1)

        self.summonGhastRandomly()


    def takeAction(self, world_state):
        '''
        Compute the next action for the agent to take.
        '''
        time1 = time.time_ns()
        # Get observation from the Minecraft world.
        player, ghasts, fireballs = self.getObservations(world_state)

        if(player==None):
            return

        # Summon ghast when there is no more.
        if len(ghasts) < 1:
            self.summonGhastRandomly()
            return

        # Parse player data.
        self.playerPos = np.array([player['x'], player['y'], player['z']])
        self.playerPitch = np.clip(player['pitch'] , -89, 89)
        self.playerYaw = -(player['yaw'] ) % 360.0 # Normalize the yaw so 0 = north.

        # Get closest ghast.
        ghast = self.getClosestEntity(self.playerPos, ghasts)
        ghastPos = np.array([ghast['x'], ghast['y'], ghast['z']])

		# Get fireball position and velocity.
        # If there is no fireball, use ghast position.
        useGhastAsfireball = len(fireballs) == 0
        if not useGhastAsfireball:
            fireball = self.getClosestEntity(self.playerPos, fireballs)
            fireballPos = np.array([fireball['x'], fireball['y'], fireball['z']])
            fireballVelocity = np.array([fireball['motionX'], fireball['motionY'], fireball['motionZ']])/.05
        else:
            fireballPos = ghastPos.copy()
            fireballVelocity = np.array([0,0,0])

        # Set the player position and rotation in the virtual world.
        self.virtualWorld.player.transform.position = self.playerPos
        self.virtualWorld.player.transform.set_rotation(self.playerPitch, self.playerYaw) 

        # Create the observation input for the NN.
        ghastPoint = self.virtualWorld.player.transform.world_to_local(ghastPos) * np.asarray([1,1,1])
        fireballPoint = self.virtualWorld.player.transform.world_to_local(fireballPos)*np.asarray([1,1,1])
        fireballVel = self.virtualWorld.player.transform.world_to_local(fireballVelocity,direction=True)* np.asarray([1,1,1])
        observationData = np.asarray([ghastPoint,fireballPoint,fireballVel]).reshape(9,1)

        # Get the output from the NN.
        cmd = self.virtualWorld.player.brain(observationData)

        # Parse the output to Malmo format.
        move = f"move {cmd[0][0] * 2 -1 }"
        strafe = f"strafe {cmd[1][0] *2-1}"
        pitch = f"pitch {cmd[2][0] *-2 + 1}"
        turn = f"turn {cmd[3][0] * 2 - 1}"
        attack = f"attack {1 if cmd[4][0] > .01 else 0}"

        # Run the output.
        self.agent_host.sendCommand(move)
        #time.sleep(0.1)
        self.agent_host.sendCommand(strafe)
        #time.sleep(0.1)
        self.agent_host.sendCommand(pitch)
        #time.sleep(0.1)
        self.agent_host.sendCommand(turn)
        #time.sleep(0.1)
        self.agent_host.sendCommand(attack)
        #time.sleep(0.05)
        time2 = time.time_ns()
        #print("Computation completed in %f"%(time2-time1))

    def getObservations(self, world_state):
        '''
        Get the player, ghasts, and fireballs from the world state.
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

        playerX, playerY, playerZ = playerPos
        closest = None
        closestDistance = -1
        for entity in entities:
            entityX, entityY, entityZ = entity['x'], entity['y'], entity['z']
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
        print(f'Summoned ghast at ({x:.2f}, {y:.2f}, {z:.2f}), yaw: {yaw:.2f}')
        time.sleep(0.5)

    def summonGhastAroundPlayer(self, degree, distance, y, stationary=True):
        '''
        Summon a Ghast at a certain degree [0,360) relative to the player.
        If degree is 0, then the ghast will be summoned in front of the player.
        If stationary, then the summoned Ghast will be inside a minecart.
        '''
        
        assert degree >= 0 and degree < 360
        degree += self.playerYaw
        x = distance * math.cos(math.radians(degree - 90))
        z = distance * math.sin(math.radians(degree - 90))
        yaw = degree if degree <= 180 else degree - 360 # Fix yaw degree for NN input.
        self.summonGhast(x + self.playerPos[0], y, z + self.playerPos[2], yaw, stationary)

    def summonGhastRandomly(self, stationary=True):
        '''
        Summon a Ghast at a random degree, distance, and height relative to the player.
        '''
        
        degree = random.randint(0, 359)
        distance = random.randint(10, 15)
        height = self.playerPos[1] + random.randint(0, 5) # If too high, ghast doesn't attack player?
        self.summonGhastAroundPlayer(degree, distance, height, stationary)




if __name__ == '__main__':
    main()
