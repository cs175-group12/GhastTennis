#the point of this module is to simulate the observations to our agent without going through malmo

#need box collider with intersect, line query, point query
#need world coordinates, transformed to local space with linear algebra for agent observations (linear algebra)
#need transform for objects 
#need world state with update tick



'''
    give observation        \\  entity update
    sample input            //
    update world
'''

'''entity info
a sword attack is a raycast from top center to 2.5 blocks away
a ghast is 4x4
a ghast fireball is 1x1

'''

import numpy as np
#import pyquaternion as pq #pip install pyquaternion if you dont have this
import time
import quaternionic as quat
deltaTime = .05



'''
reward for : fireball hitting ghast (happens in on collision)
player attacking : small negative
player hitting fireball : moderate positive, increasing to a peak the closer to hitting the fireball only rewarding early not late
facing towards ghast : tiny positive per frame, higher the closer
facing towards fireball : tiny positive per frame, higher the closer
'''

#requires testing : reset, attacking the fireball - done
#requires feature : fireball dies when it hits hte player

class world: 
    '''
    u
    '''
    def __init__(self):
        self.player = agent(self,0)
        self.entities = list()
        self.time = 0
        self.idcounter = 1
        self.score = 0.0
        #self.random = np.random.default_rng()
        self.ghastsKilled = 0
        g = ghast(self, xyz= (0,0,10) )
        #g.teleport() #testing this out
        self.spawn(g)
        return
    
    def observe(self):                                                                      #return the closest ghast and fireball to the agent
        self.closestFireball = None
        self.closestGhast = None
        for e in self.entities:
            offset = e.transform.position - self.player.transform.position
            distanceToAgent = offset.dot(offset) #technically square but doesnt matter
            df = 100000
            dg = 100000
            if type(e) is fireball:
                if(distanceToAgent < df):
                    df = distanceToAgent
                    self.closestFireball = e
            elif type(e) is ghast:
                if (distanceToAgent < dg):
                    dg = distanceToAgent
                    self.closestGhast = e
        return (self.closestGhast,self.closestFireball) #translate into the agents local space

    def start(self):
        #np.random.seed(int(time.time()))
        self.update()
        print(self.score)
        #print(self.closestGhast.transform.position)

    def update(self):
        while(self.time < 40): #going to do avg of 3 runs
            self.update_world()
            self.update_agent()
            self.update_rewards()
            self.time += deltaTime
        return
    
    def reset(self, basescore=0):
        self.time = 0
        self.idcounter = 1
        self.score = basescore
        self.player.reset()
        self.closestFireball=None
        self.closestGhast=None
        self.entities.clear()
        g = ghast(self, xyz=(0,0,10))
        self.spawn(g)
    
    def update_world(self):
        self.check_collisions()
        for e in self.entities :
            e.update()
        return

    def update_agent(self):
        self.player.update()
        return

    
    def check_collisions(self):
        e = self.entities
        for ent in e :
            #check against player
            if(np.sum((ent.transform.position - self.player.transform.position)**2) <= (ent.radius + self.player.radius)**2):
                ent.on_collision(self.player)
                self.player.on_collision(ent)
        for i in range(len(self.entities)):
            #check against other entitites
            for j in range(0,i):
                if ( np.sum((e[i].transform.position - e[j].transform.position)**2) <= (e[i].radius + e[j].radius)**2):             #simple collision check for spheres
                    e[i].on_collision(e[j])
                    e[j].on_collision(e[i])

    def raycast(self, ray): #may not be necessary
        return

    def update_rewards(self):
        self.reward_facing()
        return

    def destroy(self, entity):
        self.entities.remove(entity)
        if(entity == self.closestGhast):
            self.closestGhast = None
        if(entity == self.closestFireball):
            self.closestFireball = None

    def spawn(self,entity):
        entity.id = self.idcounter
        self.idcounter+=1
        self.entities.append(entity)

    def reward_attack(self, hit : bool, direction, fireball): #fireball may be none
        #self.score -= .1
        if(hit):
            self.score+= 5

    def reward_explode(self):#player hit by fireball
        return
    
    def reward_facing(self):
        if(self.closestFireball == None or self.closestGhast == None):
            return
        # #reward looking towards the ghast and fireball
        # fwd = self.player.transform.forward
        # pos = self.player.transform.position
        # ghastdir = (self.closestGhast.transform.position - pos)
        # ghastdir /= np.sqrt(ghastdir.dot(ghastdir))
        # fireballdir= (self.closestFireball.transform.position - pos)
        # fireballdir/= np.sqrt(fireballdir.dot(fireballdir))
        # self.score += (fwd.dot(ghastdir)**2 + fwd.dot(fireballdir)**2)*.25

        # # #punish looking at the ground and ceiling
        # fwd = self.player.transform.forward
        # pitchfactor = fwd.dot(np.asarray([0,-1,0]))**2
        # self.score -= pitchfactor * .25
        return

    def reward_fireballxghast(self):
        self.score += 50

    def prepare_pickling(self):
        self.player.transform.quaternion = np.asarray(self.player.transform.quaternion)
        for e in self.entities:
            e.transform.quaternion = np.asarray(e.transform.quaternion)

    def prepare_unpickle(self):
        self.player.transform.quaternion = quat.array(self.player.transform.quaternion)
        for e in self.entities:
            e.transform.quaternion = quat.array(e.transform.quaternion)

class transform:
    '''The transform is a component of every entity that represents its position, rotation, and scale in space'''
    def __init__(self , xyz = (0,0,0)):
        self.position = np.asarray(xyz[0:3], dtype=np.float32)
        self.quaternion = quat.array((1,0,0,0))                                                          #quaternion identity
        self.pitch = 0
        self.yaw = 0
        self.scale = np.ones((1,3) , dtype = np.float32)                                                         #scale identity
        self.forward = np.asarray([0,0,-1], dtype=np.float32)

    def get_position(self):
        return self.position.copy()

    def world_to_local(self, point, direction = False):                                                        #transforms a point from world space to local
        point = point.copy()
        if(not direction):
            point -= self.position
        #not doing scale
        return self.quaternion.inverse.rotate(point)
    
    def local_to_world(self,point, direction = False):
        point = point.copy()
        point = self.quaternion.rotate(point)
        if(not direction):
            point += self.position
        return point
        
        
    def translate(self, translation):
        self.position += translation
        return
    
    def rotate(self, dpitch, dyaw):                                                         #in minecraft, positive z is north, and 0 degrees faces north
        self.pitch += dpitch                                                                #in malmo, positive yaw goes right, negative left
        self.yaw += dyaw                                                                         #in malmo , positive pitch goes down, negative up 
        self.pitch = np.clip( self.pitch , -89, 89.0)                                                    #pitch is clamped between -90 and 90
        self.yaw += (360 if self.yaw < -.01 else 0) - (360 if self.yaw >= 360 else 0)           #yaw loops over
        fwd = np.asarray([0,0,-1], dtype = np.float32)
        
        #old pyquaternion implementation
        '''
        yawq = pq.Quaternion( axis = [0,1,0] , degrees = -self.yaw )
        leftq = pq.Quaternion( axis = [0,1,0], degrees = -self.yaw - 90)
        left = leftq.rotate(fwd)
        fwd = yawq.rotate(fwd)
        pitchq = pq.Quaternion( axis = left , degrees = self.pitch)
        fwd = pitchq.rotate(fwd)
        '''
        #new quaternionic implementation, twice as fast
        q1 = quat.array.from_axis_angle((np.deg2rad(self.pitch),0,0)) # pitch
        q2 = quat.array.from_axis_angle((0,np.deg2rad(self.yaw),0)) # yaw
        self.quaternion = q2*q1
        self.forward = self.quaternion.rotate(fwd)                        
        return
    
    def set_rotation(self, pitch, yaw):
        dpitch = pitch - self.pitch
        dyaw = yaw - self.yaw
        self.rotate(dpitch,dyaw)
    


class entity:                                                                               #base class
    '''an entity is an actor within the world, that has its update function called every "frame" by the world.
    start is used to extend its functionality so that init doesnt have to be rewritten'''
    def __init__(self,world,id=-1, xyz = (0,0,0)):
        self.transform = transform(xyz)
        self.radius = 1.0
        self.world = world
        self.id = id
        self.start()
        return

    def start(self):
        return

    def update(self):
        return

    def on_collision(self,other):
        return

class fireball(entity):
    '''the fireball is spawned by ghasts, and moves in a fixed direction until it is hit by the player or times out'''
    def start(self):
        self.velocity = np.asarray([0.0,0.0,1.0])
        self.radius = .5
        self.birthtime = self.world.time
        self.lifetime = 5
        return

    def update(self):
        self.transform.translate(self.velocity * deltaTime)
        if(self.world.time - self.birthtime > self.lifetime):
            self.world.destroy(self)                                                                        #thank you garbage collector

    def change_direction(self, newdir):
        self.velocity = newdir/np.sqrt(newdir.dot(newdir)) * 20                                                  #normalize for direction, new speed is 20 as it always is

class ghast(entity):                                                                                        #sit there and be a target, add reward when hit
    '''ghasts spawn and stay still, firing a fireball at the player every 2 seconds'''
    def start(self):
        #self.velocity = np.asarray([1.0,0.0,0.0])
        self.radius = 2
        self.fireinterval = 2
        self.lastfiretime = -2
        self.ghastsKilled = 0
        self.spawns = np.asarray([[10,2,10], [-10,2,10],[0,-5,10] , [0,5,10], [15,-2,0],  [-10,-10,10], [-10,0,-10], [0,0,0]], dtype=np.float32)
        return
    
    def update(self):
        if(self.world.time - self.lastfiretime > self.fireinterval):
            offset = self.world.player.transform.position  - self.transform.position
            direction = (offset)/np.sqrt( (offset.dot(offset))) #normalized direction from ghast to player
            f = fireball(self.world, xyz = self.transform.position.copy())               #create fireball at my position
            f.transform.translate(direction * 3)                                                                                 #offset it in shoot direction
            f.change_direction(direction)                                                                               #tell it to go that way
            self.world.spawn(f)                                                                                         #spawn it
            self.lastfiretime = self.world.time                                                                         #update last fire time
    
    def on_collision(self, other):
        if(type(other) == fireball):
            self.world.reward_fireballxghast()
            self.ghastsKilled+=1
            #also, randomize ghast position now relative to player. anything above them and within 30 blocks works. 
            #also set last fire time to 2 seconds ago.
            self.teleport()
            self.lastfiretime = self.world.time - 1.95
            #print("Hit ghast " , self.ghastsKilled)
        return
    
    def teleport(self):
        #f it lets just do it deterministically
        self.transform.position = self.spawns[self.ghastsKilled%8].copy()

        #patched to make sure ghast doesnt spawn above player
        # offset = np.random.random_sample((3)) * 60 - 30
        # offset[1] = np.abs(offset[1])
        # vertical = np.asarray([0,1,0])
        # offset /= np.sqrt(offset.dot(offset + .001))
        # while(vertical.dot(offset)>.81):
        #     offset[1] *=.7
        #     offset /= np.sqrt(offset.dot(offset))
        # offset *= 30

        # xz = (self.world.random.random((2))) * 15.0 + 10
        # sign = self.world.random.integers(low=0,high=2,size=(2))
        # sign += (sign==0) *-1
        # xz *= sign
        # y = self.world.random.random((1)) * 10
        # offset = np.asarray([xz[0],y[0],xz[1]]) 
        # self.transform.position = self.world.player.transform.position + offset

class agent(entity):                                          #max turn speed is 180 deg per second, max walk speed is 4.317 meters per second
    '''the agent is created by the world. think of this as
    the malmo wrapper for minecraft. call its set_ai function to dictate how it behaves.'''
    def start(self):
        self.radius = .5
        self.yaw = 0
        self.pitch = 0
        #cmd tuple format (move amt, strafe amt , pitch amt{-1 to 1}, yaw amt {-1 to 1}, and atk (0=false, 1=true))
        self.cmd = (0,0,0,0,0)
        self.observation = 0
        return
    
    def reset(self):
        self.radius = .5
        self.yaw = 0
        self.pitch = 0
        #cmd tuple format (move amt, strafe amt , pitch amt{-1 to 1}, yaw amt {-1 to 1}, and atk (0=false, 1=true))
        self.cmd = (0,0,0,0,0)
        self.observation = 0
        self.observationdata = None
        self.transform = transform()
    
    def set_AI(self, function):                               #give agent the function that recieves observations and makes a prediction 
        self.brain = function                              #agent returns its command function, which should be passed a command tuple

    def update(self):
        ghast,fireball = self.observation = self.world.observe() #flesh this out with agent information
        
        if(ghast==None or fireball == None): #if either are missing do nothing
            return

        self.observationData = (self.transform.world_to_local(ghast.transform.position), self.transform.world_to_local(fireball.transform.position), self.transform.world_to_local(fireball.velocity,direction=True))

        self.observationData = np.array(self.observationData).reshape((9,1)) #make sure this works later

        self.cmd = self.brain(self.observationData)

        #normalize from sigmoid to -1 to 1
        
        self.cmd = self.cmd * 2 - 1     

        self.turn(self.cmd[2], self.cmd[3])

        self.move(self.cmd[0], self.cmd[1])

        if(self.cmd[4] > 0):
            self.attack()
        
        #print("yaw is ", self.transform.yaw)
        return

    def turn(self,d_pitch, d_yaw):
        d_pitch = np.clip(d_pitch,-1,1)
        d_yaw = np.clip(d_yaw, -1, 1)
        self.transform.rotate(d_pitch * 9 , d_yaw * 9)
        return

    def move(self,forward,right):
        forward = np.clip(forward,-1.0,1.0)
        right = np.clip(right,-1.0,1.0) #BUG you'll wind up walking too fast if you dont cap speed to 4.317 combined
        dir = np.asarray([right,0,forward] , dtype=np.float32)
        dir *= 4.317*deltaTime*np.sqrt(dir.dot(dir))/(2**1.5)  #normalize movement direction , adjust speed to be at most 4.317
        yawq = quat.array.from_axis_angle((0,np.deg2rad(self.transform.yaw),0)) # yaw
        self.transform.translate( yawq.rotate(dir))              #cant use a simple projection or youd move weird
        return

    def attack(self):
        
        if(self.observation[1] is not None):
            fireball = self.observation[1]
            hit = False
            #see if attack ray hit fireball
            if(SphereLineIntersect(self.transform.position + self.transform.forward*fireball.radius, self.transform.position + self.transform.forward*2.5 , fireball.transform.position, fireball.radius)):
                if(fireball.velocity.dot(self.transform.forward) < 0 ):
                    fireball.change_direction(self.transform.forward)
                    hit = True
                #print("Fireball HIT!") things are hitting really often??
            self.world.reward_attack(hit,self.transform.forward, fireball)
        return
    
    def on_collision(self, other):
        if(type(other) == fireball):
            self.world.destroy(other) # fireballs blow up when they hit player
            self.world.reward_explode()

def testAI(observations):
    print(observations[1])
    return (0,0,0,0,0)

def SphereLineIntersect(pointA, pointB, center, radius):
    # [-b +- sqrt ( b**2 -4*a*c) ]/2a
    #polynomial is [ sum{(ac)**2}- r**2]  + sum{2*bau*ac} + sum{bau**2}
    try:
        c = np.sum( (pointA-center)**2 ) - radius**2
        b = np.sum(2*(pointB-pointA)*(pointA-center))
        a = np.sum((pointB-pointA)**2) 
        sol1 = (-b + np.sqrt(b**2 - 4*a*c))/ (2*a)
        sol2 = (-b - np.sqrt(b**2 - 4*a*c))/ (2*a)
    except RuntimeWarning:
        return False #imaginary square root means no valid collision
    # #for debugging
    # x = pointA + (pointB-pointA)*sol1 - center
    # y = (pointB-pointA ) * sol1 
    # check1 = np.dot(x,x)  - np.dot(y,y)

    # x = pointA + (pointB-pointA)*sol2 - center
    # y = (pointB-pointA ) * sol2 
    # check2 = np.dot(x,x)  - np.dot(y,y) 

    # print("This should be 0 if there is an intersection: " , check1)
    # print("This could also be 0 if there is an intersection: " , check2)

    if( 0 <= sol1 <= 1 or  0 <= sol2 <= 1):
        return True
    else:
        return False

#good gradient for training would be closeness to correct timing in hitting the fireball, closeness to correct angle to hit the ghast
#another good bonus would just be proximity in aiming at either the fireball or the ghast
#another small bonus just for swinging the sword perhaps? or a negative

#pass
def test1():
    pointA = np.asarray((0,0,-2))
    pointB = np.asarray((0,0,1))
    center = np.asarray((0,0,0))
    radius = .5
    x = SphereLineIntersect(pointA,pointB,center,radius)
    print("The value of x is: " , x)

def test2():
    sekai = world()
    sekai.player.set_AI(testAI)
    f = fireball(sekai, xyz= (0,10,0))
    sekai.spawn(f)
    g = ghast(sekai, xyz= (0,10,10) )
    sekai.spawn(g)
    sekai.start()

def testAI2(observations):
    print(observations[0])
    return(1,0,0,0,0)

def test3():
    sekai = world()
    sekai.player.set_AI(testAI2)
    f = fireball(sekai, xyz= (0,10,0))
    sekai.spawn(f)
    g = ghast(sekai, xyz= (0,10,10) )
    sekai.spawn(g)
    sekai.start()

def testAI3(observations):
    #print(observations[0])
    return(1,0,0,1,0)

def test4():
    sekai = world()
    sekai.player.set_AI(testAI3)

    g = ghast(sekai, xyz= (0,10,10) )
    sekai.spawn(g)
    sekai.start()

#for sphere line intersect. works !
def test5():
    grid = np.zeros( (10,10,3) )
    for i in range(10):
        for j in range(10):
            grid[i,j,0] = float(i)/10.0 - .5
            grid[i,j,1] = 1
            grid[i,j,2] = float(j)/10.0 - .5
    ray = np.asarray([0,-9,0])
    center= np.asarray([0,0,0])
    radius = .5
    screen = []
    for i in range(10):
        screen.append([])
        for j in range(10):
            screen[i].append(SphereLineIntersect(grid[i,j,:],grid[i,j,:] + ray, center, radius ))
    for i in range(10):
        print(list(map( lambda b: b*1, screen[i]) ))
    
import neuralnetdebug as nn
def test6():
    n = nn.NetworkV3([1])
    n.loadtxt(118)
    sekai = world()
    sekai.player.set_AI(n.predict)
    sekai.start()

def test7():
    n = nn.PerfectNetwork()
    sekai = world()
    sekai.player.set_AI(n.predict)
    sekai.start()

def test8():
    t = transform()
    t.set_rotation(0,90)
    print(t.forward)
    t.set_rotation(0,180)
    print(t.forward)
    t.set_rotation(0,270)
    print(t.forward)

def test9():
    t = transform()
    print(t.forward)
    t.set_rotation(0,0)
    print(t.forward)
    t.set_rotation(-90,0)
    print(t.forward)
    t.set_rotation(90,0)
    print(t.forward)

if(__name__ == "__main__"):
    starttime = time.time()
    #test4()
    #test6()
    test7()
    elapsedtime = time.time()-starttime
    print("End time is " , elapsedtime)

