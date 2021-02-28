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
import pyquaternion as pq #pip install pyquaternion if you dont have this

deltaTime = .05

class world: 
    def __init__(self):
        self.player = agent(self,0)
        self.entities = list()
        self.time = 0
        self.idcounter = 1
        return
    
    def observe(self):                                                                      #return the closest ghast and fireball to the agent
        return                                                                              #translated into the agents local space

    def update(self):
        self.update_world()
        self.update_agent()
        self.time += deltaTime
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
                if ( np.sum((e[i].position - e[j].position)**2) <= (e[i].radius + e[j].radius)**2):             #simple collision check for spheres
                    e[i].on_collision(e[j])
                    e[j].on_collision(e[i])

    def get_rewards(self):
        return

    def destroy(self, entity):
        self.entities.remove(entity)

    def spawn(self,entity):
        entity.id = self.idcounter
        self.idcounter+=1
        self.entities.append(entity)

class transform:
    def __init__(self):
        self.position = np.zeros((1,3))
        self.quaternion = pq.Quaternion()                                                              #quaternion identity
        self.pitch = 0
        self.yaw = 0
        self.scale = np.ones((1,3))                                                         #scale identity
        self.forward = np.asarray([0,0,1])

    def get_position(self):
        return self.position.copy()

    def world_to_local(self, point):                                                        #transforms a point from world space to local
        point -= self.position
        #not doing scale
        return self.quaternion.inverse.rotate(point)
        
    def translate(self, translation):
        self.position += translation
        return
    
    def rotate(self, dpitch, dyaw):                                                         #in minecraft, positive z is north, and 0 degrees faces north
        self.pitch += dpitch                                                                     #in malmo, positive yaw goes right, negative left
        self.yaw += dyaw                                                                         #in malmo , positive pitch goes down, negative up 
        fwd = np.asarray([0,0,1])
        yawq = pq.Quaternion( axis = [0,1,0] , degrees = -self.yaw )
        leftq = pq.Quaternion( axis = [0,1,0], degrees = -self.yaw - 90)
        left = leftq.rotate(fwd)
        fwd = yawq.rotate(fwd)
        pitchq = pq.Quaternion( axis = left , degrees = self.pitch)
        fwd = pitchq.rotate(fwd)
        self.forward = fwd
        self.quaternion = pitchq* yawq                                    
        return


class entity:                                                                               #base class
    def __init__(self,world,id=-1,x=0,y=0,z=0):
        self.transform = transform()
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
        self.velocity = np.asarray([0,0,1])
        self.radius = .5
        self.birthtime = self.world.time
        self.lifetime = 5
        return

    def update(self):
        self.transform.position += self.velocity * deltaTime
        if(self.world.time - self.birthtime > self.lifetime):
            self.world.destroy(self)                                                                        #thank you garbage collector

    def change_direction(self, newdir):
        self.velocity = newdir/np.sum(newdir) * 20                                                          #normalize for direction, new speed is 20 as it always is

class ghast(entity):                                                                                        #sit there and be a target, add reward when hit
    def start(self):
        self.radius = 2
        self.fireinterval = 2
        self.lastfiretime = -2
        return
    
    def update(self):
        if(self.world.time - self.lastfiretime > self.fireinterval):
            direction = (self.world.player.position - self.position)/np.sum(self.world.player.position - self.position) #normalized direction from ghast to player
            f = fireball(self.world,x=self.position[0,0] , y =self.position[0,1], z = self.position[0,2])               #create fireball at my position
            f.position += direction * 3                                                                                 #offset it in shoot direction
            f.change_direction(direction)                                                                               #tell it to go that way
            self.world.spawn(f)                                                                                         #spawn it
            self.lastfiretime = self.world.time                                                                         #update last fire time

class agent(entity):                                          #max turn speed is 180 deg per second, max walk speed is 4.317 meters per second
    def start(self):
        self.radius = .5
        self.yaw = 0
        self.pitch = 0
        self.cmd = (0,0,0,0,0)
        return
    
    def set_AI(self, function):                               #give agent the function that recieves observations and makes a prediction 
        self.function = function                              #agent returns its command function, which should be passed a command tuple

    def update(self):
        o = self.world.get_observation

        #cmd tuple format (move amt, strafe amt, yaw amt {-1 to 1}, pitch amt{-1 to 1}, and atk (0=false, 1=true))
        self.cmd = self.function(o)     



        return

    def turn(self,d_pitch, d_yaw):
        return

    def move(self,foward,right):
        return

    def attack(self):
        return
    


#good gradient for training would be closeness to correct timing in hitting the fireball, closeness to correct angle to hit the ghast
#another good bonus would just be proximity in aiming at either the fireball or the ghast
#another small bonus just for swinging the sword perhaps? or a negative