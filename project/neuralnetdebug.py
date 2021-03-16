import numpy as np
import matplotlib.pyplot as pyplot
from scipy.special import expit,logit
#takes a numpy array of values and returns a numpy array of the same length. 
def Softmax(inputs):
    return np.exp(inputs)/np.sum(inputs)

def RelU(input):
    return np.max(input,0)

def LeakyRelU(input): #maybe dont use this? More of a demonstration of what should be done inline
    #return input * .01 if input < 0 else input
    # return 1.0/(1+np.exp(-input)) #testing sigmoid out
    return expit(input)     #apparently this is faster
    #return  np.nan_to_num(    np.clip(  ((input < 0) & (1)) * .99 * input + .01 * input , a_min=-10, a_max=10  ) )
    
def Sigmoid(input):
    return expit(input)

def LeakyRelUDeriv(input):
    #return .01 if input < 0 else 1
    #return 1 - .99 * (input < 0) #true one
    return input*(1.0-input) #sigmoid test

class NetworkV2:
    def __init__(self, layersizes, learningrate = .01):
        self.neurons = list()           #list of row vectors
        self.axons = list()             #matrix connect neurons[i] to neurons[i+1]. dimension is layersizes[i] , layersizes[i+1]
        self.layersizes = layersizes    #the count of neurons in each layer
        self.biases = list()            #bonus connection to each neuron in each layer. same shape as self.neurons
        self.learningrate = .01
        self.debuginfo = {"axon deltas" : {}}
        for i in range(len(layersizes)):
            self.neurons.append(np.zeros( (1,layersizes[i]) ))
            #self.biases.append( np.random.rand(1,layersizes[i])/ (self.layersizes[i]) )        #commented for parity with video
            self.biases.append(np.zeros((1,layersizes[i])))
            #self.biases.append(np.random.rand(1,layersizes[i])/layersizes[i])
            if(i>0):
                self.axons.append(np.ones( (self.layersizes[i-1],self.layersizes[i]) ) / (self.layersizes[i-1]) )
                #self.axons.append(np.random.rand( self.layersizes[i-1],self.layersizes[i] ) / (self.layersizes[i-1]) - 1.0/(2*(self.layersizes[i-1])) ) negative-positive range start vals
                self.debuginfo["axon deltas"][i-1] = list()
        #debug ifo
        print("layersizes length %d" %(len(layersizes)))
        print("neurons length %d" %(len(self.neurons)) )
        print("axons length %d" %(len(self.axons)) )
        print("biases length %d" %(len(self.biases)) )

    def predict(self, input):
        '''
        def predict(self, input):
        input should be a numpy array of shape 1,inputsize. remember to normalize input to 0-1 range
        '''
        np.copyto(self.neurons[0], input)
        for i in range(1,len(self.layersizes)):
            self.neurons[i] = LeakyRelU(self.neurons[i-1] @ self.axons[i-1] + self.biases[i])       #nk km nm
            
            self.neurons[i] = np.nan_to_num(self.neurons[i],posinf=10,neginf=-10)                       #clear nan
        return np.copy(self.neurons[-1])

    def backpropagate(self,errors):
        '''def backpropagate(self,errors):'''
        np.copyto(self.neurons[-1], errors)
        for i in range(len(self.layersizes)-1,0,-1 ):
            depth = len(self.layersizes)-i
            axonupdate = np.transpose(-self.learningrate * (depth**2) * self.neurons[i-1]) @ self.neurons[i]
            self.debuginfo["axon deltas"][i-1].append(np.sum(np.abs(axonupdate)))
            self.axons[i-1] += axonupdate               #nk km nm  so Trans(neurons[i-1] ) @ neurons[i] 
            self.biases[i] += -self.learningrate * (depth**2) * self.neurons[i]                                     #update biases
            deriv  = LeakyRelUDeriv(self.neurons[i-1])                                                              #get derivative of this layer of neurons  after inverting sigmoid
            self.neurons[i-1] = deriv * (self.neurons[i] @ np.transpose(self.axons[i-1]))                           #backpropagate errors into previous layer
            
            self.axons[i-1] = np.nan_to_num(self.axons[i-1], posinf=10, neginf=-10)                                 #get rid of weird values.
            self.biases[i-1] = np.nan_to_num(self.biases[i-1], posinf=10, neginf=-10)
            self.neurons[i-1] = np.nan_to_num(self.neurons[i-1], posinf=.9999, neginf=-.9999)
        return
    
    def train(self,vals,labels,epochs = 1, testvals = None, testlabels=None):
        '''
        def train(self,vals,labels):
        gradient descent styled training. labels should be indexes of the output that are "correct" 
        '''
        originallearningrate = self.learningrate

        if(testvals is not None and testlabels is not None):
            self.debuginfo["learning rate"] = list()

        for e in range(epochs):
            #self.learningrate/=1.5
            for i in range(vals.shape[0]):
                if i%100 == 0:
                    print("\rTraining started. %2.0f %% done."%(float(i + e*vals.shape[0])/float(vals.shape[0]*epochs) * 100), end='\r')
                #print(i)
                #p = Softmax(self.predict(vals[i]))         commented for parity with video
                p = self.predict(vals[i])
                correct = np.zeros((1,self.layersizes[-1]))
                correct[0,labels[i]] = 1
                #correct = LeakyRelU(correct)                 #used to make softmax more effective, now makes it worse.
                err = p - correct                        
                #err = np.nan_to_num(err, posinf=10, neginf=-10)            commented for parity with video
                self.backpropagate(err)
            if(testvals is not None and testlabels is not None):
                self.debuginfo["learning rate"].append(self.test(testvals,testlabels,False))
        self.learningrate = originallearningrate
        print("Training complete")
    
    def test(self,vals,labels, printout=True):
        numcorrect= 0
        total = 0
        
        for i in range(vals.shape[0]):
            if i%100 == 0 and printout:
                print("\rTest started. %20.0f %% done."%(float(i)/vals.shape[0] * 100), end='\r')
            if np.argmax(self.predict(vals[i:i+1,:]))==labels[i]:
                numcorrect+=1
            total+=1
        if printout:
            print("Test Complete. %d out of %d correct." %(numcorrect, total))
        return float(numcorrect)/total

import copy

#todo : do biases
class NetworkV3:
    #previously axons were size (input,output) and input was a row vector. output computed as input*axons + biases
    #now for performance we flip everything. axons are (output,input) input is a column vector, output computed as axons*input
    #nk km nm   11 input 10 output  10,11   11, 1   10,1
    def __init__(self, layersizes, learningrate = .01):
        self.neurons = list()           #list of row vectors
        self.axons = list()             #matrix connect neurons[i] to neurons[i+1]. dimension is layersizes[i] , layersizes[i+1]
        self.layersizes = layersizes    #the count of neurons in each layer
        self.biases = list()            #bonus connection to each neuron in each layer. same shape as self.neurons
        for i in range(len(layersizes)):
            self.neurons.append(np.zeros( (layersizes[i],1) ))
            self.biases.append((np.random.rand(layersizes[i],1)-.5)*2)#)/layersizes[i])
            if(i>0):
                self.axons.append( (np.random.rand( self.layersizes[i],self.layersizes[i-1] ) -.5)*2  ) # / (self.layersizes[i-1]) - 1.0/(2*(self.layersizes[i-1])) ) 
        return

    def reproduce(self):
        child = NetworkV3([])
        child.neurons = copy.deepcopy(self.neurons)
        child.axons = copy.deepcopy(self.axons)
        child.layersizes = copy.deepcopy(self.layersizes)
        child.biases = copy.deepcopy(self.biases)
        return child

    def mutate(self, rate, amount = 1):
        for i in range(len(self.axons)):
            delta = np.random.rand(self.axons[i].shape[0], self.axons[i].shape[1])
            gamma = (np.random.rand(self.axons[i].shape[0], self.axons[i].shape[1]) -.5) * amount
            delta = (delta < rate) / self.axons[i].shape[1]*gamma
            self.axons[i] += delta  
            delta_b = np.random.rand(self.biases[i+1].shape[0], self.biases[i+1].shape[1])
            gamma_b = (np.random.rand(self.biases[i+1].shape[0], self.biases[i+1].shape[1]) -.5 ) * amount
            delta_b = (delta_b < rate) / self.biases[i].shape[0]*gamma_b
            self.biases[i+1] += delta_b

    def predict(self, input):
        '''
        def predict(self, input):
        input should be a numpy array of shape inputsize,1 . 
        '''
        np.copyto(self.neurons[0], input)
        for i in range(1,len(self.layersizes)):
            self.neurons[i] = Sigmoid(  self.axons[i-1] @ self.neurons[i-1] + self.biases[i])       #nk km nm
        return np.copy(self.neurons[-1])
    
    def savetxt(self,r : int):
        for i in range(1,len(self.layersizes)):
            np.savetxt("networks/biases%d_%d.csv"%(r,i) , self.biases[i])
            np.savetxt("networks/weights%d_%d.csv"%(r,i), self.axons[i-1])
    
    def save(self,r : int):
        #use npy later
        self.savetxt(r)



    def loadtxt(self,r:int):
        biases = [np.zeros((9,1))]
        weights = []
        try:
            for i in range(1,10):
                biases.append(np.loadtxt("networks/biases%d_%d.csv"%(r,i) , dtype=np.float32).reshape(-1,1))
                weights.append(np.loadtxt("networks/weights%d_%d.csv"%(r,i), dtype=np.float32))        
        except OSError:
            pass
        self.biases= biases
        self.axons = weights
        self.layersizes = [] # didnt save layersizes so im just assuming input it 9
        self.neurons.clear() #technically i could get that from shape[1] of weights[0] tho
        #self.neurons.append()
        for i in range(len(self.biases)):
            self.layersizes.append(len(self.biases[i]))
            self.neurons.append(np.zeros((len(self.biases[i]),1)))


class PerfectNetwork():
    def __init__(self):
        self.neurons = list()           #list of row vectors
        self.axons = list()             #matrix connect neurons[i] to neurons[i+1]. dimension is layersizes[i] , layersizes[i+1]
        self.layersizes = layersizes = [9,5]    #the count of neurons in each layer
        self.biases = list()            #bonus connection to each neuron in each layer. same shape as self.neurons
        for i in range(len(layersizes)):
            self.neurons.append(np.zeros( (layersizes[i],1) ))
            self.biases.append(np.zeros((layersizes[i],1)))
        self.axons.append(np.asarray(
            [[0,0,1,0,0,0,0,0,0],
            [0,0,0,.5,0,0,0,0,0],
            [0,-1,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,-1,0,0,0]] , dtype = np.float32))
        self.biases[1][4,0] = 2.5
        return

    def predict(self, input):
        '''
        def predict(self, input):
        input should be a numpy array of shape inputsize,1 . remember to normalize input to 0-1 range
        '''
        np.copyto(self.neurons[0], input)
        for i in range(1,len(self.layersizes)):
            self.neurons[i] = Sigmoid(  self.axons[i-1] @ self.neurons[i-1] + self.biases[i])       #nk km nm
        return np.copy(self.neurons[-1])

#wrapper for network v3 to be used in network v4
#takes in a mask for input (of same dimension) and outputs 1
class Subnetwork():
    #mask must be an array of 1's and 0's
    def __init__(self,mask):
        input_0_size = len(mask)
        self.inputsize = np.sum(mask)
        self.mask = mask

    
class NeuralNetV4():
    def __init__(self,inputsize,outputsize):
        self.networks = []
        

    
    #basic flow of program
    #create a list of tuples of world , network pairs
    #run all the worlds
    #sort the list by the world scores
    #remove the bottom 128
    #x = 65, for i to x reproduce network, mutate, x/=2
    #store highscore and continue









def main():
    # mnimg = mnist.train_images().reshape(60000,28**2)/256.0 + 1.0/256.0
    # mnlabel = mnist.train_labels()
    # network2 = NetworkV2([28**2,10], learningrate =.02)
    # network2.train(mnimg[0:2000,:],mnlabel[0:2000], epochs=3, testvals=mnimg[11000:12000,:] , testlabels=mnlabel[11000:12000] )
    # network2.test(mnimg[12000:13000,:] , mnlabel[12000:13000])
    # pyplot.plot(network2.debuginfo["axon deltas"][0] , color= "red")
    # pyplot.show()
    # pyplot.plot(network2.debuginfo["learning rate"] , color= "blue")
    # pyplot.show()
    # return
    x = NetworkV3([1])
    x.loadtxt(7)
    #x.mutate()
    b = 44



if(__name__== "__main__"):
    main()
