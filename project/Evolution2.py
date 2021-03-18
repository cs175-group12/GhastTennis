from neuralnetdebug import NeuralNetV4
import notminecraft
import threading
import numpy as np
import matplotlib.pyplot as pyplot
import concurrent.futures as futures
import multiprocessing as mp
import warnings
import os
warnings.filterwarnings("ignore")

#multiprocessing.set_start_method("spawn",True)
population = 128 #power of 2, population *3/4 must be divisible by threads
boom = 8    #large population will be 8x size of population
generations = 250
saveas = 6
mutation_factor = 3
AIsAndWorlds = list()
threads = 12

highscores = list()
#ideas for improvement : set a new random seed in reset each time, run 3 samples and average scores
#boom bust population size : introduct catastrophes and boons
#mix up mutation rate
#ideally : huge first generation, like gargantuan, then trim it down to just the top 128 or so and balloon them all up
#pretrain for 10 generations too
#maybe instead of log by placement just softmax
#since score , against what I've been attempting to do, seems deterministic...
#technically I dont need to rerun the ones that have already run right?

'''how about this
i maintain a "hall of fame" between generations. lets say size 128
the first generation gets evaluated, sorted, and inserted into hall of fame

wait in my current situation wouldn't that be the same as if only the bottom half of the 
list got evaluated? or actually the bottom 3/4 need reevaluation
'''
errcount = 0

args = []






def main():
    #setup contexts for multiprocessing
    m = mp.Manager()
    Q = m.Queue()
    for i in range(0, threads):
        args.append([i*(int(population/threads)),(i+1)*(int(population/threads)),Q,AIsAndWorlds])
    #initialize random gen 1
    for i in range(population):
        n = NeuralNetV4(9,5,0,1,9,10,p_mask = .111)
        AIsAndWorlds.append([n, notminecraft.world()])
        AIsAndWorlds[i][1].prepare_pickling() #prepare pickling unnecessary with no processes
    
    #load last best due to error failure
    # loadlast = NeuralNetV4.load(saveas-1)
    # AIsAndWorlds[0][0] = loadlast
    # AIsAndWorlds[0][1].player.set_AI(loadlast.predict)

    for g in range(generations):

        #evaluate (can be threaded)
        Eval(0,population,Q)
        #for multithread pool, dont reevaluate static top quarter
        if(g==1):
            args.clear()
            for i in range(0, threads):
                args.append([i*(int(population/threads*3/4)),(i+1)*(int(population/threads*3/4)),Q,AIsAndWorlds])
        
        #sort the array, store highscore
        AIsAndWorlds.sort(key = lambda w : w[1].score)
        highscores.append(AIsAndWorlds[-1][1].score)

        sexual_selection()
        
        print("Generation %d : %f "%(g,highscores[g]))
    
    print("Highscore: ", AIsAndWorlds[-1][1].score)
    AIsAndWorlds[-1][0].save(saveas)
    AIsAndWorlds[-2][0].save(saveas+1)
    AIsAndWorlds[-3][0].save(saveas+2)
    np.save("Highscores/Highscore_%d,_generations_%d,_population_%d,mutation_factor_%d" % (saveas,generations,population,mutation_factor), np.asarray(highscores))

        

    pyplot.plot(highscores)
    pyplot.show()

    return
#Q is a managed multithreading queue, if you want to post into Ais and worlds from there
def Evaluate(first,last, Q = None):
    errcount = 0
    try:
        with mp.Pool(processes=threads) as pool:
            pool.starmap(run_worlds,args)
    except Exception as e:
        print(e)
        errcount += 1
        if(errcount>=3):
            AIsAndWorlds[-1][0].save(saveas*-1)

    #new step from multiprocessing, unpack Q and assign scores
    while(not Q.empty()):
        i,score = Q.get()
        AIsAndWorlds[i][1].score = score

Eval = Evaluate
#use that go call evaluate globally

def ReplaceAI(dest, src):
    AIsAndWorlds[dest][0] = AIsAndWorlds[src][0].reproduce()
    AIsAndWorlds[dest][0].mutate(np.random.rand(1)*.9, mutation_factor)
    AIsAndWorlds[dest][1].player.set_AI(AIsAndWorlds[dest][0].predict)

def ReplaceAIWith(dest, ai):
    AIsAndWorlds[dest][0] = ai
    AIsAndWorlds[dest][0].mutate(np.random.rand(1)*.9, mutation_factor)
    AIsAndWorlds[dest][1].player.set_AI(AIsAndWorlds[dest][0].predict)

def RandomizeAI(dest):
    n = NeuralNetV4(9,5,0,1,9,10,p_mask = .111)
    AIsAndWorlds[dest][0] = n
    AIsAndWorlds[dest][1].player.set_AI(AIsAndWorlds[dest][0].predict)
    


def sexual_selection():
    newai = []
    deviances = np.random.randint(-population//8,0, population//2 )
    for i in range( int(population//4*3), population):
        for c in range(2):
            newai.append(AIsAndWorlds[i][0].reproduce_with(AIsAndWorlds[i + deviances[(i-int(population//4*3))*2 + c  ] ][0]))
    for i in range(int(population//2)):
        ReplaceAIWith(i,newai[i])
    for i in range(population//2,population//8*5):
        RandomizeAI(i)
    for i in range(population//8*5, population//8*6):
        AIsAndWorlds[i][0].mutate(np.random.rand()*.9,mutation_factor)

#with pop 128, bottom 64 deleted, 16 are randomized, 16 mutate in place, the remainder reproduce according to the scale
#Replace the worst half of the items with reproductions from the top quarter, using log2 distribution
#this is the log2 distribution used for reproduction
reproductionScale = []
for i in range(0,population):
    reproductionScale.append(int(np.floor(-np.log2( (population-i)/population )) -1))
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
        #for o in range(3):
        AIsAndWorlds[i][1].reset(basescore=0)
        AIsAndWorlds[i][1].start()
        messages.append([i,AIsAndWorlds[i][1].score]) #for multiprocessing
    for i in range(len(messages)):
        Q.put(messages[i])




if(__name__ == "__main__" ): 
    main()
    
