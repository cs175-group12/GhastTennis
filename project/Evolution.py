from neuralnetdebug import NetworkV3
import notminecraft
import threading
import numpy as np
import matplotlib.pyplot as pyplot
import concurrent.futures as futures
AIsAndWorlds = list()
highscores = list()

def main():

    generations = 100
    population = 32
    threads = 4
    
    #initialize random gen 1
    for i in range(population):
        layersizes = np.random.randint(low = 9, high = 81, size = np.random.random_integers(3,6))
        layersizes[0] = 9   #inputsize
        layersizes[-1] = 5  #outputsize
        n = NetworkV3(layersizes)
        AIsAndWorlds.append([n, notminecraft.world()])
    
    for g in range(generations):

        #evaluate (can be threaded)
        #run_worlds(0,population)
        

        args = [[] ,[]]
        for i in range(0, threads):
            args[0].append(i*int(population/threads))
            args[1].append(i*int(population/threads)+int(population/threads))

        with futures.ThreadPoolExecutor(max_workers=threads) as executor:
            executor.map(run_worlds, args[0], args[1])

        #sort the array, store highscore
        AIsAndWorlds.sort( key = lambda w : w[1].score)
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
    print(AIsAndWorlds[-1][0].save(1))

        

    pyplot.plot(highscores)
    pyplot.show()

    return
    

def run_worlds(a,b):
    for i in range(a,b):
        AIsAndWorlds[i][1].player.set_AI(AIsAndWorlds[i][0].predict) 
        AIsAndWorlds[i][1].reset()
        AIsAndWorlds[i][1].start()
        print("running world " , i)  



if(__name__ == "__main__" ): 
    main()