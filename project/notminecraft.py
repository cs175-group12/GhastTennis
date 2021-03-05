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

class world: 
    def __init__(self):
        self.player = agent(self,0)
        self.entities = list()
        self.time = 0
        self.idcounter = 1
        return
    
    def observe(self):                                                                      #return the closest ghast and fireball to the agent
        closestFireball = None
        closestGhast = None
        for e in self.entities:
            offset = e.transform.position - self.player.transform.position
            distanceToAgent = offset.dot(offset) #technically square but doesnt matter
            df = 100000
            dg = 100000
            if type(e) is fireball:
                if(distanceToAgent < df):
                    df = distanceToAgent
                    closestFireball = e
            elif type(e) is ghast:
                if (distanceToAgent < dg):
                    dg = distanceToAgent
                    closestGhast = e
        return (closestGhast,closestFireball) #translated into the agents local space

    def start(self):
        self.update()

    def update(self):
        while(self.time < 100):
            self.update_world()
            self.update_agent()
            self.time += deltaTime
            if(self.time > 2.65):
                x = 3
        return
    
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
        for i in range(len(self.entities)):
            for j in range(0,i):
                if ( np.sum((e[i].transform.position - e[j].transform.position)**2) <= (e[i].radius + e[j].radius)**2):             #simple collision check for spheres
                    e[i].on_collision(e[j])
                    e[j].on_collision(e[i])

    def raycast(self, ray): #may not be necessary
        return

    def get_rewards(self):
        return

    def destroy(self, entity):
        self.entities.remove(entity)

    def spawn(self,entity):
        entity.id = self.idcounter
        self.idcounter+=1
        self.entities.append(entity)

class transform:
    def __init__(self , xyz = (0,0,0)):
        self.position = np.asarray(xyz[0:3], dtype=np.float32)
        self.quaternion = quat.array((1,0,0,0))                                                          #quaternion identity
        self.pitch = 0
        self.yaw = 0
        self.scale = np.ones((1,3) , dtype = np.float32)                                                         #scale identity
        self.forward = np.asarray([0,0,1], dtype=np.float32)

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
        np.clip( self.pitch , -89, 89.0)                                                    #pitch is clamped between -90 and 90
        self.yaw += (360 if self.yaw < -.01 else 0) - (360 if self.yaw >= 360 else 0)           #yaw loops over
        fwd = np.asarray([0,0,1], dtype = np.float32)
        
        #old pyquaternion implementation
        '''
        yawq = pq.Quaternion( axis = [0,1,0] , degrees = -self.yaw )
        leftq = pq.Quaternion( axis = [0,1,0], degrees = -self.yaw - 90)
        left = leftq.rotate(fwd)
        fwd = yawq.rotate(fwd)
        pitchq = pq.Quaternion( axis = left , degrees = self.pitch)
        fwd = pitchq.rotate(fwd)
        '''
        #new quaternionic implementation
        q1 = quat.array.from_axis_angle((np.deg2rad(self.pitch),0,0)) # pitch
        q2 = quat.array.from_axis_angle((0,np.deg2rad(self.yaw),0)) # yaw
        self.quaternion = q2*q1
        self.forward = self.quaternion.rotate(fwd)                        
        return


class entity:                                                                               #base class
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
    def start(self):
        self.radius = 2
        self.fireinterval = 2
        self.lastfiretime = -2
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

class agent(entity):                                          #max turn speed is 180 deg per second, max walk speed is 4.317 meters per second
    def start(self):
        self.radius = .5
        self.yaw = 0
        self.pitch = 0
        #cmd tuple format (move amt, strafe amt , pitch amt{-1 to 1}, yaw amt {-1 to 1}, and atk (0=false, 1=true))
        self.cmd = (0,0,0,0,0)
        self.observation = 0
        return
    
    def set_AI(self, function):                               #give agent the function that recieves observations and makes a prediction 
        self.brain = function                              #agent returns its command function, which should be passed a command tuple

    def update(self):
        ghast,fireball = self.world.observe() #flesh this out with agent information
        
        if(ghast==None or fireball == None): #if either are missing do nothing
            return

        self.observation= (self.transform.world_to_local(ghast.transform.position), self.transform.world_to_local(fireball.transform.position), self.transform.world_to_local(fireball.velocity,direction=True))

        self.cmd = self.brain(self.observation)     

        self.turn(self.cmd[2], self.cmd[3])

        self.move(self.cmd[0], self.cmd[1])

        if(self.cmd[4] != 0):
            self.attack()
        
        #print("Time is ", self.world.time)
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
        fireball = self.observation[1]
        if(fireball != None):
            #secant line test
            if(SphereLineIntersect(self.transform.position, self.transform.position + self.transform.forward*2.5 , fireball.transform.position, fireball.radius)):
                fireball.change_direction(self.transform.forward)
                print("Fireball HIT!")
        return

def testAI(observations):
    print(observations[1])
    return (0,0,0,0,0)

def SphereLineIntersect(pointA, pointB, center, radius):
    # [-b +- sqrt ( b**2 -4*a*c) ]/2a
    #polynomial is [ sum{(ac)**2}- r**2]  + sum{2*bau*ac} + sum{bau**2}
    c = np.sum( (pointA-center)**2 ) - radius**2
    b = np.sum(2*(pointB-pointA)*(pointA-center))
    a = np.sum((pointB-pointA)**2) 
    sol1 = (-b + np.sqrt(b**2 - 4*a*c))/ (2*a)
    sol2 = (-b - np.sqrt(b**2 - 4*a*c))/ (2*a)
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
    f = fireball(sekai, xyz= (0,10,0))
    sekai.spawn(f)
    g = ghast(sekai, xyz= (0,10,10) )
    sekai.spawn(g)
    sekai.start()

if(__name__ == "__main__"):
    starttime = time.time()
    test4()
    elapsedtime = time.time()-starttime
    print("End time is " , elapsedtime)

