# %% Import the relevant packages needed for the script
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import shutil

# %% Inputs
# NOTATION OF DV = [rle xup xlo zup zlo zxxup zxxlo dste zte betate alphate]
# Some testing cases for initialisation comment/uncomment
#lowerbound = [0.005,0.1,0.1,0.02,0.02,0.3,0.3,1e-4,-0.2,5,0]
#upperbound = [0.1,0.4,0.4,0.1,0.1,0.6,0.6,1e-2,0.2,15,10]
lowerbound = [1.1019* (0.05)**2,0.1,0.1,0.05,0.05,(0.4),(0.4),0.68e-3,0,5,0]
upperbound = [1.1019* (0.20)**2,0.4,0.4,0.1,0.1,(0.5),(0.5),4.68e-3,0,20,5]
lenChromo = 11              #The length of our chromosome
RankLenght = 4              #How many top performer chromosomes are selected from each generation
splitpos1 = 4               #The split point of the 2 out of 4 selected chromosomes to be cross-overed
splitpos2 = 6               #The split point of the other 2 out of 4 selected chromosomes to be cross-overed
nonmutatedL = 8             #Length of non-mutated population 
PopSize = 2 * nonmutatedL   #When we mutate our total population size becomes twice the non-mutated population
population = []             #Initialize the list of our population
Iterlimit = 100             #Generation limit
bestPerf = []               # Store the best performers
bestPerfLD =[]              # Track the score of the best performer of each generation

# %% Functions to be used throughout 

def convert(row,lowerbound,upperbound):
    #This function will take in a random number and convert it to an absolute value within the bounds
    abspop = []
    for count in range(len(row)):
        abspop.append(lowerbound[count] + row[count]*(upperbound[count]-lowerbound[count]))
    return abspop

def parsec(abspop,filename):
    # our design variables
    rle = abspop[0]
    xup = abspop[1]
    xlo = abspop[2] 
    zup = abspop[3]
    zlo = abspop[4]
    zxxup = -abspop[5]
    zxxlo = -abspop[6]
    dzte = abspop[7]
    zte = abspop[8]
    betate = abspop[9] * np.pi / 180
    alphate = abspop[10] * np.pi / 180
    
    # Upper surface
    # find the coefficients for the UPPER SURFACE
    # https://www.researchgate.net/publication/268574488_A_Comparison_of_Airfoil_Shape_Parameterization_Techniques_for_Early_Design_Optimization
    xte =1
    a = np.array([[1, 0, 0, 0, 0, 0],
                  [xte**0.5,xte**1.5,xte**2.5,xte**3.5,xte**4.5,xte**5.5],
                  [xup**0.5,xup**1.5,xup**2.5,xup**3.5,xup**4.5,xup**5.5],
                  [0.5*xte**-0.5,1.5*xte**0.5,2.5*xte**1.5,3.5*xte**2.5,4.5*xte**3.5,5.5*xte**4.5],
                  [0.5*xup**-0.5,1.5*xup**0.5,2.5*xup**1.5,3.5*xup**2.5,4.5*xup**3.5,5.5*xup**4.5],
                  [-0.25*xup**-1.5,0.75*xup**-0.5,(15/4)*xup**0.5,(35/4)*xup**1.5,(53/4)*xup**2.5,(99/4)*xup**3.5]
                  ])
    
    b = np.array([np.sqrt(2*rle),
                  zte + 0.5*dzte,
                  zup,
                  np.tan(alphate-betate),
                  0,
                  zxxup])
    
    # solve to find the coefficients
    coeffs = np.linalg.solve(a, b)
    x = np.linspace(0.00,1,400)
    yupper = np.zeros(len(x))
    for count in range(6):
        yupper = yupper + ( coeffs[count] * x**(count+1-0.5))
    # Write the upper surface file
    fid=open(filename+".dat","w")
    for count in range(len(x)):
        fid.write(str(x[len(x)-count-1]) + '  ' + str(yupper[len(x)-count-1]) +'\n')
    # Find the coefficients for the lower surface
    a = np.array([[1, 0, 0, 0, 0, 0],
              [xte**0.5,xte**1.5,xte**2.5,xte**3.5,xte**4.5,xte**5.5],
              [xlo**0.5,xlo**1.5,xlo**2.5,xlo**3.5,xlo**4.5,xlo**5.5],
              [0.5*xte**-0.5,1.5*xte**0.5,2.5*xte**1.5,3.5*xte**2.5,4.5*xte**3.5,5.5*xte**4.5],
              [0.5*xlo**-0.5,1.5*xlo**0.5,2.5*xlo**1.5,3.5*xlo**2.5,4.5*xlo**3.5,5.5*xlo**4.5],
              [-0.25*xlo**-1.5,0.75*xlo**-0.5,(15/4)*xlo**0.5,(35/4)*xlo**1.5,(53/4)*xlo**2.5,(99/4)*xlo**3.5]
              ])
    
    b = np.array([np.sqrt(2*rle),
                  zte + 0.5*dzte,
                  zlo,
                  np.tan(alphate-betate),
                  0,
                  zxxlo])
    
    # solve to find the coefficients
    coeffs = np.linalg.solve(a, b)
    # initialise for speed
    ybottom = np.zeros(len(x))
    # Find the values of the bottom surface
    for count in range(6):
        ybottom = ybottom + ( coeffs[count] * x**(count+1-0.5))
    
    #Write the bottom surface file
    for count in range(len(x)):
        fid.write(str(x[count]) + '  ' + str(-ybottom[count]) +'\n')
    fid.close()
    
def xfoilRun(airfoilName,polarname):
    fileID = open("commands.txt","w")
    fileID.write("load " + airfoilName + ".dat \n")
    #Required for unknown files
    fileID.write(airfoilName + ".dat \n")
    # Add the pane line for smoothing
    fileID.write("PANE \n")
    
    fileID.write("OPER \n")
    fileID.write("VPAR\n")
    fileID.write("N 5\n")
    fileID.write(" \n")
    fileID.write("visc 10e5\n")
    fileID.write("PACC\n")
    fileID.write(polarname + ".txt\n")
    fileID.write(" \n")
    fileID.write("aseq 0 15 1\n")
    fileID.write(" \n")
    fileID.write("quit\n")
    fileID.write("exit\n")
    fileID.close()
    os.system("xfoil.exe < commands.txt")
    return 

def readLD(polarname):
    # good check but sometimes it might create a polar with empty details...
    if os.path.isfile(polarname + ".txt"):
        fid = open(polarname + ".txt","r")
        lines = fid.readlines()
        fid.close()
        lines = lines[12:]
        if len(lines)==0:   # ...This will take care of the existing but non converged file
            return 0
        for count in range(len(lines)):
            lines[count] = lines[count].split()
            lines[count] = [float(x) for x in lines[count]]
        data = np.array(lines)
        LD = data[:,1]/data[:,2]
        return max(LD)
    else:
        return 0

def select(topNum,popdict):
    #This function selects the top performers from a dictionary of form {score:chromosome,...}
    tempdict = (sorted(popdict.items(), key=lambda item: item[0],reverse=True)[:topNum])
    return tempdict

def crossover(chromosome1,chromosome2,splitpos):
    # Performs a crossover for two chromosomes based on a prescribed split position
    cross1 = np.append(chromosome1[:splitpos],chromosome2[splitpos:])
    cross2 = np.append(chromosome2[:splitpos],chromosome1[splitpos:])
    return np.array(cross1), np.array(cross2)

def mutate(chromosome):
    #Performs a mutation for each of the non-mutated element
    Muidx = random.randint(0,lenChromo-1)
    chromosome[Muidx] = np.random.rand()
    return chromosome

# %%
#########################   ENTRY POINT     #########################
    
# Initialize our population
population = np.random.rand(nonmutatedL,lenChromo)

# Move to scoring 
gennum = 0
countChromo=-1

# used for evaluation
scores = []
popdict = {}
for row in population:
    # Convert it to absolute values
    abspop = convert(row,lowerbound,upperbound)
    # Create a new airfoil
    countChromo+=1
    filename = "airfoil" + str(gennum) + str(countChromo)
    parsec(abspop,filename)
    # Run XFOIL
    polarname= "polar" + str(gennum) + str(countChromo)
    xfoilRun(filename,polarname)
    # Extract maximum L/D from all angles
    data = readLD(polarname)
    
    # Evaluate the fitness function for our initial random population
    scores.append(readLD(polarname))
    popdict[scores[-1]]=population[countChromo,:]
    
# Select the top 4 performers
tempdict = select(RankLenght,popdict)
selectedChromo=[]
for count in range(RankLenght):
    selectedChromo.append(tempdict[count][1])

# Store it for monitoring purposes
bestPerf.append(selectedChromo[0])   
bestPerfLD.append(max(scores))

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
    selectedChromo[count] = mutate(selectedChromo[count])

# Repetition
gennum+=1
countChromo=-1
population = np.array(selectedChromo.copy())
for row in population:
    # Convert it to absolute values
    abspop = convert(row,lowerbound,upperbound)
    # Create a new airfoil
    countChromo+=1
    filename = "airfoil" + str(gennum) + str(countChromo)
    parsec(abspop,filename)
    # Run XFOIL
    polarname= "polar" + str(gennum) + str(countChromo)
    xfoilRun(filename,polarname)
    # Extract maximum L/D from all angles
    data = readLD(polarname)
    
    # Evaluate the fitness function for our initial random population
    scores.append(readLD(polarname))
    popdict[scores[-1]]=population[countChromo,:]

iterationcounter = 1 
while 1:
    print(iterationcounter,bestPerfLD[-1])
    iterationcounter+=1
    # Select the top 4 performers
    tempdict = select(RankLenght,popdict)
    selectedChromo=[]
    for count in range(RankLenght):
        selectedChromo.append(tempdict[count][1])
    
    # Store it for monitoring purposes
    bestPerf.append(selectedChromo[0])   
    bestPerfLD.append(max(scores))
    
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
        selectedChromo[count] = mutate(selectedChromo[count])
    
    # Repetition
    gennum+=1
    countChromo=-1
    population = np.array(selectedChromo.copy())
    for row in population:
        # Convert it to absolute values
        abspop = convert(row,lowerbound,upperbound)
        # Create a new airfoil
        countChromo+=1
        filename = "airfoil" + str(gennum) + str(countChromo)
        parsec(abspop,filename)
        # Run XFOIL
        polarname= "polar" + str(gennum) + str(countChromo)
        xfoilRun(filename,polarname)
        # Extract maximum L/D from all angles 
        data = readLD(polarname)
        
        # Evaluate the fitness function for our population
        scores.append(readLD(polarname))
        popdict[scores[-1]]=population[countChromo,:]
    if iterationcounter == Iterlimit:
        break

# %% Export results
## Convert to dataframe
df = pd.DataFrame(data=[bestPerf,bestPerfLD])
df.head()
df.to_csv('Finals.csv')

# %% Organise the output in a new folder
# Create a new folder
os.mkdir("Results")
sourcepath = os.getcwd()
sourcefiles = os.listdir(sourcepath)

# All our result files are csv, txt and dat files only
destination = sourcepath + "\\Results"

for file in sourcefiles:
    if file.endswith('.csv') or file.endswith('.txt') or file.endswith('.dat'):
        shutil.move(os.path.join(sourcepath,file), os.path.join(destination,file))
        
# %% Visualising Results

# Visualise score propagation through generations
plt.plot(range(Iterlimit),bestPerfLD)
plt.show()

# Get the final shape and plot it   
final = convert(bestPerf[-1],lowerbound,upperbound)
parsec(final,"final")
fid = open("final.dat","r")
lines = fid.readlines()
fid.close()
xval=[]
yval=[]
for line in lines:
    line = line.split()
    xval.append(float(line[0]))
    yval.append(float(line[1]))

plt.plot(xval,yval,'xk')
plt.show()


                                     
                                   


