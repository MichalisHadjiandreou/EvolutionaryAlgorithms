"""
Created on Tue Mar  9 23:35:41 2021

@author: micha
"""

## Import the relevant packages
import numpy as np
import pandas as pd
import random

## Inputs
lowerbound = -10            #Lower bound of our search domain
upperbound = 10             #Upper bound of our search domain
lenChromo = 6               #The length of our chromosome
exactresult = -36           #The exact answer of our equation
RankLenght = 4              #How many top performer chromosomes are selected from each generation
splitpos1 = 2               #The split point of the 2 out of 4 selected chromosomes to be cross-overed
splitpos2 = 3               #The split point of the other 2 out of 4 selected chromosomes to be cross-overed
nonmutatedL = 8             #Length of non-mutated population 
PopSize = 2 * nonmutatedL   #When we mutate our total population size becomes twice the non-mutated population
population = []             #Initialize the list of our population
Iterlimit = 100000          #Generation limit
bestPerf = []               # Store the best performers

## Functions to be used throughout 

def randomize(lowerbound,upperbound,lenChromo):
    #This function is used just once when we initialize our population
    randomlist = []                 
    for count in range(lenChromo):
        randomlist.append(random.randint(lowerbound,upperbound))
    return randomlist    


def chromoEval(arr):
    #This is the evaluation of our equation
    return 5*arr[0]-3*arr[1]+6*arr[2]+arr[3]-7*arr[4]+2*arr[5]

def score(arr,exactresult):
    #This is the evaluation of our fitness function
    return abs(chromoEval(arr)-exactresult)

def select(topNum,popdict):
    #This function selects the top performers from a dictionary of form {score:chromosome,...}
    tempdict = (sorted(popdict.items(), key=lambda item: item[0],reverse=False)[:topNum])
    return tempdict

def crossover(chromosome1,chromosome2,splitpos):
    # Performs a crossover for two chromosomes based on a prescribed split position
    cross1 = np.append(chromosome1[:splitpos],chromosome2[splitpos:])
    cross2 = np.append(chromosome2[:splitpos],chromosome1[splitpos:])
    return np.array(cross1), np.array(cross2)

def mutate(chromosome,lowerbound,upperbound):
    #Performs a mutation for each of the non-mutated element
    Muidx = random.randint(0,lenChromo-1)
    chromosome[Muidx] = random.randint(lowerbound,upperbound)
    return chromosome


#########################   ENTRY POINT     #########################
    
# Initialize our population
for count in range(nonmutatedL):
    population.append(randomize(lowerbound,upperbound,lenChromo))
population = np.array(population)


# Evaluate the fitness function for our initial random population
scores = []
popdict = {}
for count in range(len(population)):
    scores.append(score(population[count,:],exactresult))
    popdict[scores[-1]]=population[count,:]
    
# Select the top performers
tempdict = select(RankLenght,popdict)
selectedChromo=[]
for count in range(RankLenght):
    selectedChromo.append(tempdict[count][1])

# Store it for monitoring purposes
bestPerf.append(selectedChromo[0])   

# Crossover
firstpairIndex = random.sample(range(0, RankLenght), 2)
secondpairIndex = [x for x in range(0,RankLenght) if x not in firstpairIndex]

cross1,cross2 = crossover(selectedChromo[firstpairIndex[0]],selectedChromo[firstpairIndex[-1]],splitpos1)
cross3,cross4 = crossover(selectedChromo[secondpairIndex[0]],selectedChromo[secondpairIndex[-1]],splitpos2)

# Add the crossovers on our population
selectedChromo.append(cross1)
selectedChromo.append(cross2)
selectedChromo.append(cross3)
selectedChromo.append(cross4)

# Mutate
for count in range(0,len(selectedChromo)):
    selectedChromo[count] = mutate(selectedChromo[count],lowerbound,upperbound)
    
population = np.array(selectedChromo.copy())

# Evaluate the fitness function for our initial random population
scores = []
popdict = {}
for count in range(len(population)):
    scores.append(score(population[count,:],exactresult))
    popdict[scores[-1]]=population[count,:]

counter = 1

## Repeat until convergence
if 0 in scores:
    # Store it for monitoring purposes
    bestPerf.append(popdict[0])
    print(popdict[0])
else:
    while 1:
        counter+=1
        print(counter)
            # Select the top performers
        tempdict = select(RankLenght,popdict)
        selectedChromo=[]
        for count in range(RankLenght):
            selectedChromo.append(tempdict[count][1])
        
        # Store it for monitoring purposes
        bestPerf.append(selectedChromo[0])

        firstpairIndex = random.sample(range(0, RankLenght), 2)
        secondpairIndex = [x for x in range(0,RankLenght) if x not in firstpairIndex]
        
        cross1,cross2 = crossover(selectedChromo[firstpairIndex[0]],selectedChromo[firstpairIndex[-1]],splitpos1)
        cross3,cross4 = crossover(selectedChromo[secondpairIndex[0]],selectedChromo[secondpairIndex[-1]],splitpos2)
        
        # Add the crossovers on our population
        selectedChromo.append(cross1)
        selectedChromo.append(cross2)
        selectedChromo.append(cross3)
        selectedChromo.append(cross4)
        
        # Mutate
        for count in range(0,len(selectedChromo)):
            selectedChromo[count] = mutate(selectedChromo[count],lowerbound,upperbound)
            
        population = np.array(selectedChromo.copy())
        # Evaluate the fitness function for our initial random population
        scores = []
        popdict = {}
        for count in range(len(population)):
            scores.append(score(population[count,:],exactresult))
            popdict[scores[-1]]=population[count,:]
        
        if 0 in scores or counter==Iterlimit:
            break
        
## Conver to dataframe
df = pd.DataFrame(data=bestPerf)
df.head()
df.to_csv('Finals.csv')