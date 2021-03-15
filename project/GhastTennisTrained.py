import sys
import time
import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import MalmoPython
import notminecraft
from neuralnetdebug import NetworkV3, PerfectNetwork

def main():
    # Load NN data.
    trainedAI = PerfectNetwork()
    #trainedAI.loadtxt(119) # load trained agent from files

    # Create agent.
    runs = 10
    agent = Agent(trainedAI)
    for _ in range(runs):
        agent.start()



class Agent():
    def __init__(self, trainedAI):
        # Agent Parameters
        self.trainedAI = trainedAI
        self.virtualWorld = None

        # Malmo Host
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

    def start(self):
        """
        Start the agent.
        """

        # Initialize Malmo and the Minecraft world.
        world_state = self.initMalmo()
        self.initWorld()

        # Initialize virtual world for the NN.
        self.virtualWorld = notminecraft.world()
        self.virtualWorld.player.set_AI(self.trainedAI.predict)
        time.sleep(1)

        # Take action while the mission is running.
        while world_state.is_mission_running:
            print(".", end="")
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            self.takeAction(world_state)
            for error in world_state.errors:
                print("Error:", error.text)

        # End mission.
        print()
        print("Mission ended")

    def initMalmo(self):
        """
        Initialize new Malmo mission.
        """

        # Load the XML file and create mission spec & record.
        print("Initializing...")
        mission_file = './mission.xml'
        with open(mission_file, 'r') as f:
            print("Loading mission from %s" % mission_file)
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
        return world_state

    def initWorld(self):
        '''
        Initialize the Minecraft world.
        Clean the world, make the player invincible, and spawn ghasts.
        '''

        self.cleanWorld()
        self.makeInvincible()
        time.sleep(0.1)

        # Summon a ghast randomly in front of the player.
        # x = random.randint(-10, 10)
        # self.summonGhast(x, 3, -20)

        # Summon a ghast randomly around the player.
        degree = random.randint(-180, 179)
        self.summonGhastAroundPlayer(degree, 20, 3)

    def takeAction(self, world_state):
        '''
        Compute the next action for the agent to take.
        '''

        # Get observation from the Minecraft world.
        player, ghasts, fireballs = self.getObservations(world_state)

        #if(len(ghasts) == 0 or len(fireballs) == 0): #if either are missing do nothing
        #    return

        if len(ghasts) == 0:
            return
        useghastasfireball = False
        if len(fireballs) == 0:
            useghastasfireball = True

        # Parse player data.
        playerPos = np.array([player['x'], player['y'], player['z']])
        pitch = np.clip(player['pitch'] , -89, 89)
        yaw = -(player['yaw'] +180 ) %360.0

        # Get closest ghast and fireball.
        ghast = self.getClosestEntity(playerPos, ghasts)
        #fireball = self.getClosestEntity(playerPos, fireballs)

		    # Get ghast position and fireball position & velocity.
        ghastPos = np.array([ghast['x'], ghast['y'], ghast['z']])
        fireball = 0 
        fireballPos = 0
        fireballVelocity = 0
        if useghastasfireball == False:
            fireball = self.getClosestEntity(playerPos, fireballs)
            fireballPos = np.array([fireball['x'], fireball['y'], fireball['z']])
            fireballVelocity = np.array([fireball['motionX'], fireball['motionY'], fireball['motionZ']])
        else:
            fireballPos = ghastPos.copy()
            fireballVelocity = np.array([0,0,0])

        # Set the player position and rotation in the virtual world.
        self.virtualWorld.player.transform.position = playerPos
        self.virtualWorld.player.transform.set_rotation(pitch,yaw) 

        # Create the observation input for the NN.
        ghastPoint = self.virtualWorld.player.transform.world_to_local(ghastPos)
        fireballPoint = self.virtualWorld.player.transform.world_to_local(fireballPos)
        fireballVel = self.virtualWorld.player.transform.world_to_local(fireballVelocity,direction=True)
        observationData = np.asarray([ghastPoint,fireballPoint,fireballVel]).reshape(9,1)

        # Get the output from the NN.
        cmd = self.virtualWorld.player.brain(observationData)

        # Parse the output to Malmo format.
        move = f"move {cmd[0][0] * 2 -1 }"
        strafe = f"strafe {cmd[1][0] * 2 - 1}"
        pitch = f"pitch {cmd[2][0] *-2 + 1}"
        turn = f"turn {cmd[3][0] * 2 - 1}"
        attack = f"attack {1 if cmd[4][0] > 0 else 0}"

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
        time.sleep(0.05)

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

    def summonGhastAroundPlayer(self, degree, distance, y, stationary=True):
        '''
        Summon a Ghast at a certain degree [-180,180) relative to the player.
        Assume the player is facing north and is at x=0, z=0.
        If stationary, then the summoned Ghast will be inside a minecart.
        '''
        
        assert (degree >= -180 and degree < 180)
        x = distance * math.cos(math.radians(degree - 270))
        z = distance * math.sin(math.radians(degree - 270))
        yaw = degree
        self.summonGhast(x, y, z, yaw, stationary)



if __name__ == '__main__':
    main()
