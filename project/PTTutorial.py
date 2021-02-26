#pytorch tutorial

import numpy as np
import matplotlib.pyplot as pyplot
import mnist
import torch

x = torch.empty(2,2,3)             #multi dimensional tensor. it seems that the last dimension is the most inner nested.
print(x,"\n")
x = x + x                          #doubles x
x.add_(x)                          #doubles x, but in place operation. these have an _ at the end
print(x.view(12))                  #a reshaped x
print()
print(x.view(-1,2))                #figure out automatically how many rows we have

b = np.ones(5)
y = torch.from_numpy(b)            #can convert into torch from numpy

'''gpu = torch.device("cuda")'''            #cuda isnt enabled on my system so this doesnt work
'''y.to(gpu)'''                          #send to gpu
#perform operations like normal
'''y.to("cpu")'''                        #retrieve from gpu
y.numpy()                          #convert to numpy array

#end tutorial two

#tutorial 3
#nothing worth noting

#tutorial 4
'''ok so pytorch helps with backpropagation of arbitrary functions by 
tracking the functions you apply to a tracked variable, then deriving 
and chainruling your final output relative to that tracked input.'''
#full example

x=torch.tensor(1.0)
y=torch.tensor(2.0)
w=torch.tensor(1.0,requires_grad=True)
yhat=w*x
loss = (yhat-y)**2
print(loss)


loss.backward()                 #backprop
print(w.grad)

#update weight

#tutorial 5

class tut5:

    def __init__(self):
        self.x = np.array([1,2,3,4], dtype = np.float32)
        self.y = np.array([2,4,6,8], dtype = np.float32)
        self.w = 0  
        return

    def fwd(self,x):
        return self.w*x

    def loss(self,y,yhat):
        return ((y-yhat)**2).mean()

    #accomodates vectors, derivative of MSE is 2x*error
    def gradient(self,x,y,yhat):
        return np.dot(2*x,yhat-y).mean()

    def train(self):

        print("Init prediction : f(5) = %1.1f"%(self.fwd(5)))
        #train
        learningrate = .01
        n_iters = 10
        for epoch in range(n_iters):
            y_pred = self.fwd(self.x)
            L = self.loss(self.y,y_pred)
            dwdx = self.gradient(self.x,self.y,y_pred)
            self.w -= learningrate*dwdx
            if epoch % 2 == 0 : 
                print("Epoch %d : w = %1.1f , loss = %1.1f"%(epoch,self.w,L))
        return
    
example5 = tut5()
example5.train()

#example 2 uses autograd for backpropagation, which makes it shitter but more general and easier to code.
class tut52:

    def __init__(self):
        self.x = torch.tensor([1,2,3,4], dtype = torch.float32)
        self.y = torch.tensor([2,4,6,8], dtype = torch.float32)
        self.w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        return

    def fwd(self,x):
        return self.w*x

    def loss(self,y,yhat):
        return ((y-yhat)**2).mean()

    # unnecessary now
    # def gradient(self,x,y,yhat):
    #     return np.dot(2*x,yhat-y).mean()

    def train(self):

        print("Init prediction : f(5) = %1.1f"%(self.fwd(5)))
        #train
        learningrate = .01
        n_iters = 10
        for epoch in range(n_iters):
            y_pred = self.fwd(self.x)
            L = self.loss(self.y,y_pred)
            L.backward()                            #automatically backpropagates, stores value into w.grad
            with torch.no_grad():
                self.w -= learningrate*self.w.grad       
            self.w.grad.zero_()
            if epoch % 2 == 0 : 
                print("Epoch %d : w = %1.1f , loss = %1.1f"%(epoch,self.w,L))
        return
    
example52 = tut52()
example52.train()

 #tutorial 6
 