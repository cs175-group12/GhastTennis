from neuralnetdebug import NetworkV3
import notminecraft
import threading
import numpy as np
import matplotlib.pyplot as pyplot

AIsAndWorlds = list()
highscores = list()

def main():

    generations = 100
    population = 32

    #initialize random gen 1
    for i in range(population):
        layersizes = np.random.randint(low = 9, high = 81, size = np.random.random_integers(3,6))
        layersizes[0] = 9   #inputsize
        layersizes[-1] = 5  #outputsize
        n = NetworkV3(layersizes)
        AIsAndWorlds.append([n, notminecraft.world()])
    
    for g in range(generations):

        #evaluate (can be threaded)
        for i in range(population):
            AIsAndWorlds[i][1].player.set_AI(AIsAndWorlds[i][0].predict) 
            AIsAndWorlds[i][1].reset()
            AIsAndWorlds[i][1].start()
            print("running world " , i)

        #sort the array, store highscore
        AIsAndWorlds.sort( key = lambda w : w[1].score)
        highscores.append(AIsAndWorlds[-1][1].score)

        #cut off bottom half
        #generating new ai can be multithreaded (also id like a different scheme for this)
        for i in range(int(population/2)):
            AIsAndWorlds[i][0] = AIsAndWorlds[population-1-i][0].reproduce()
            AIsAndWorlds[population-1-i][0].mutate(.1) #pick a random number or something for this later
            AIsAndWorlds[population-1-i][1].player.set_AI(AIsAndWorlds[population-1-i][0].predict)
        
        print(highscores[g])
    
    print("Highscore: ", AIsAndWorlds[-1][1].score)
    print(AIsAndWorlds[-1][0].save(0))

        

    pyplot.plot(highscores)
    pyplot.show()

    return
    

    



if(__name__ == "__main__" ): 
    main()