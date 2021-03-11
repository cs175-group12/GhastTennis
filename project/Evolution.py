from neuralnetdebug import NetworkV3
import notminecraft
import threading
import numpy as np
import matplotlib.pyplot as pyplot
import concurrent.futures as futures
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

#multiprocessing.set_start_method("spawn",True)
population = 16 #power of 2
generations = 200

AIsAndWorlds = list()

#ideas for improvement : set a new random seed in reset each time, run 3 samples and average scores
#boom bust population size : introduct catastrophes and boons
#mix up mutation rate
#ideally : huge first generation, like gargantuan, then trim it down to just the top 128 or so and balloon them all up
#pretrain for 10 generations too
#make the ghasts move significantly
#have score be avg of n runs with randomized ghast position

def main():

    highscores = list()
    threads = 8
    m = mp.Manager()
    Q = m.Queue()

    #initialize random gen 1
    for i in range(population):
        layersizes = np.random.randint(low = 9, high = 81, size = np.random.random_integers(3,6))
        layersizes[0] = 9   #inputsize
        layersizes[-1] = 5  #outputsize
        n = NetworkV3(layersizes)
        AIsAndWorlds.append([n, notminecraft.world()])
        AIsAndWorlds[i][1].prepare_pickling()
    
    args = []
        
    for i in range(0, threads):
        args.append([i*(int(population/threads)),i*(int(population/threads))+(int(population/threads)),Q,AIsAndWorlds])

    for g in range(generations):

        #evaluate (can be threaded)
        
        with mp.Pool(processes=threads) as pool:
            pool.starmap(run_worlds,args)


        #new step from multiprocessing, unpack Q and assign scores
        while(not Q.empty()):
            i,score = Q.get()
            AIsAndWorlds[i][1].score = score
        
        #sort the array, store highscore
        AIsAndWorlds.sort(key = lambda w : w[1].score)
        highscores.append(AIsAndWorlds[-1][1].score)

        natural_selection()
        #cut off bottom half
        #generating new ai can be multithreaded (also id like a different scheme for this)
        
        print(highscores[g])
    
    print("Highscore: ", AIsAndWorlds[-1][1].score)
    print(AIsAndWorlds[-1][0].save(7))

        

    pyplot.plot(highscores)
    pyplot.show()

    return

def ReplaceAI(dest, src):
    AIsAndWorlds[dest][0] = AIsAndWorlds[src][0].reproduce()
    AIsAndWorlds[dest][0].mutate(np.random.rand(1)*.9)
    AIsAndWorlds[dest][1].player.set_AI(AIsAndWorlds[dest][0].predict)

def RandomizeAI(dest):
    layersizes = np.random.randint(low = 9, high = 81, size = np.random.random_integers(3,6))
    layersizes[0] = 9   #inputsize
    layersizes[-1] = 5  #outputsize
    n = NetworkV3(layersizes)
    AIsAndWorlds[dest][0] = n
    AIsAndWorlds[dest][1].player.set_AI(AIsAndWorlds[dest][0].predict)
    
#this is the log2 distribution used for reproduction
reproductionScale = []
for i in range(0,population):
    reproductionScale.append(int(np.floor(-np.log2( (population-i)/population )) -1))

#with pop 128, 64 deleted, 16 are randomized, 16 mutate in place, the remainder reproduce according to the scale
#Replace the worst half of the items with reproductions from the top quarter, using log2 distribution
def natural_selection():
    i=0
    for k in range(population-1,-1,-1):
        if(reproductionScale[k] <1):
            break
        for g in range(reproductionScale[k]):
            ReplaceAI(g+i, k)
        i+=reproductionScale[k]
    
   #items that dont reproduce or die mutate, or are randomized
    q= int(i/4)
    for k in range(i,i+q):
        RandomizeAI(k)
    for k in range(i+q,i+2*q):
        AIsAndWorlds[k][0].mutate(np.random.rand(1)*.9)
    return

def natural_selection_beta():
    for i in range(int(population/4)):
        AIsAndWorlds[i*2][0] = AIsAndWorlds[population-1-i][0].reproduce()
        AIsAndWorlds[i*2+1][0] = AIsAndWorlds[population-1-i][0].reproduce()
        AIsAndWorlds[i*2][0].mutate(.1) #pick a random number or something for this later
        AIsAndWorlds[i*2+1][0].mutate(.1) #pick a random number or something for this later
        AIsAndWorlds[i*2][1].player.set_AI(AIsAndWorlds[i*2][0].predict)
        AIsAndWorlds[i*2+1][1].player.set_AI(AIsAndWorlds[i*2+1][0].predict)
    

def run_worlds(a,b,Q : mp.Queue, AIsAndWorlds):
    messages = []
    for i in range(a,b):
        AIsAndWorlds[i][1].prepare_unpickle()
        AIsAndWorlds[i][1].player.set_AI(AIsAndWorlds[i][0].predict) 
        AIsAndWorlds[i][1].score =0
        for o in range(3):
            AIsAndWorlds[i][1].reset(basescore=AIsAndWorlds[i][1].score)
            AIsAndWorlds[i][1].start()
        messages.append([i,AIsAndWorlds[i][1].score]) #for multiprocessing
    for i in range(len(messages)):
        Q.put(messages[i])




if(__name__ == "__main__" ): 
    main()
    
