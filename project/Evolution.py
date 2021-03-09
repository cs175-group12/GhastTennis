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

def main():

    AIsAndWorlds = list()
    highscores = list()
    generations = 2000
    population = 32
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
        args.append([i*int(population/threads),i*int(population/threads)+int(population/threads),Q,AIsAndWorlds])

    for g in range(generations):

        #evaluate (can be threaded)
        #run_worlds(0,population)
        
        with mp.Pool(processes=threads) as pool:
            pool.starmap(run_worlds,args)


        #new step from multiprocessing, unpack Q and assign scores
        while(not Q.empty()):
            i,score = Q.get()
            AIsAndWorlds[i][1].score = score
        
        #sort the array, store highscore
        AIsAndWorlds.sort(key = lambda w : w[1].score)
        highscores.append(AIsAndWorlds[-1][1].score)

        #cut off bottom half
        #generating new ai can be multithreaded (also id like a different scheme for this)
        for i in range(int(population/4)):
            AIsAndWorlds[i*2][0] = AIsAndWorlds[population-1-i][0].reproduce()
            AIsAndWorlds[i*2+1][0] = AIsAndWorlds[population-1-i][0].reproduce()
            AIsAndWorlds[i*2][0].mutate(.1) #pick a random number or something for this later
            AIsAndWorlds[i*2+1][0].mutate(.1) #pick a random number or something for this later
            AIsAndWorlds[i*2][1].player.set_AI(AIsAndWorlds[i*2][0].predict)
            AIsAndWorlds[i*2+1][1].player.set_AI(AIsAndWorlds[i*2+1][0].predict)
        
        print(highscores[g])
    
    print("Highscore: ", AIsAndWorlds[-1][1].score)
    print(AIsAndWorlds[-1][0].save(3))

        

    pyplot.plot(highscores)
    pyplot.show()

    return
    

def run_worlds(a,b,Q : mp.Queue, AIsAndWorlds):
    messages = []
    for i in range(a,b):
        AIsAndWorlds[i][1].prepare_unpickle()
        AIsAndWorlds[i][1].player.set_AI(AIsAndWorlds[i][0].predict) 
        AIsAndWorlds[i][1].reset()
        AIsAndWorlds[i][1].start()
        messages.append([i,AIsAndWorlds[i][1].score]) #for multiprocessing
    for i in range(len(messages)):
        Q.put(messages[i])




if(__name__ == "__main__" ): 
    main()