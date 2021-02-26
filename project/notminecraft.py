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

deltaTime = .05

class world: 
    def __init__(self):
        self.player = agent(self,0)
        self.entities = list()
        self.time = 0
        self.idcounter = 1
        return
    
    def observe(self):
        return

    def update(self):
        self.update_world()
        self.update_agent()
        self.time += deltaTime
        return
    
    def update_world(self):
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

class entity:
    def __init__(self,world,id=-1,x=0,y=0,z=0):
        self.transform = np.identity((4,4))
        self.position = np.zeros((1,3))
        self.quaternion = np.asarray([0,0,0,1]) #quaternion identity
        self.scale = np.ones((1,3)) #scale identity
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
        self.position += self.velocity * deltaTime
        if(self.world.time - self.birthtime > self.lifetime):
            self.world.destroy(self)                                     #thank you garbage collector

    def change_direction(self, newdir):
        self.velocity = newdir/np.sum(newdir) * 20      #normalize for direction, new speed is 20 as it always is

class ghast(entity):                        #sit there and be a target, add reward when hit
    def start(self):
        self.radius = 2
        self.fireinterval = 2
        self.lastfiretime = -2
        return
    
    def update(self):
        if(self.world.time - self.lastfiretime > self.fireinterval):
            direction = (self.world.player.position - self.position)/np.sum(self.world.player.position - self.position) #normalized direction
            f = fireball(self.world,x=self.position[0,0] , y =self.position[0,1], z = self.position[0,2])
            f.position += direction * 3
            f.change_direction(direction)
            self.world.spawn(f)
            self.lastfiretime = self.world.time

class agent(entity):
    def start(self):
        self.radius = .5
        return

#good gradient for training would be closeness to correct timing in hitting the fireball, closeness to correct angle to hit the ghast
#another good bonus would just be proximity in aiming at either the fireball or the ghast
#another small bonus just for swinging the sword perhaps? or a negative