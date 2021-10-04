import functools
from PIL.Image import new
import numpy as np
import functools
import operator
import cv2
import matplotlib.pyplot as plt
import itertools
import random

def makeImageSmall(img_array):
    return cv2.resize(img_array, (600, 300))

def img2chromo(img_array):
    return np.resize(img_array, (functools.reduce(operator.mul, img_array.shape)))

def chromo2img(chromo, img_shape):
    return np.resize(chromo, img_shape)

def initialPopulation(img_shape, num_individuals = 8):
    # Creating the initial population. Multiple different chromosomes are create(8 in this case), and the same number stays throughout the algo
    #the reason 8 different chromosomes have been taken is that now, we can take the best few number as parents, and use them to create a new, better, generation
    initPopulation = np.empty(shape = (num_individuals, functools.reduce(operator.mul, img_shape)), dtype = np.uint8)
    for ind_num in range(num_individuals):
        initPopulation[ind_num] = np.random.random(functools.reduce(operator.mul, img_shape))*256
    return initPopulation

def fitness(target_chrom, current_chrom):
    quality = np.mean(np.abs(target_chrom - current_chrom))
    quality = np.sum(target_chrom) - quality
    return quality

def totalFitness(target_chrom, solutions):
    #gets the overall fitness of the chromosome in question
    qualities = np.zeros(solutions.shape[0])
    for solution in range(solutions.shape[0]):
        qualities[solution] = fitness(target_chrom, solutions[solution])
    return qualities

def getParents(solutions, qualities, numParents = 3):
    #taking best chromosomes among the list of chromosomes provided. 
    parents = np.empty((numParents, solutions.shape[1]), dtype= np.uint8)
    for indParent in range(numParents):
        maxIndex = np.where(qualities == np.max(qualities))
        maxIndex = maxIndex[0][0]
        parents[indParent] = solutions[maxIndex]
        qualities[maxIndex] = -1
    return parents

def crossover(parents, img_shape, numIndividuals = 8):
    newPopulation = np.empty(shape = (numIndividuals, functools.reduce(operator.mul, img_shape)) , dtype = np.uint8)
    newPopulation[0:parents.shape[0]] = parents
    newGenToMake = numIndividuals - parents.shape[0]
    parents_permutations = list(itertools.permutations(iterable= np.arange(0, parents.shape[0]), r = 2))
    selected_permutations = random.sample(range(len(parents_permutations)), newGenToMake)
    combIndex = parents.shape[0] #if the number of parents is decided this can be skipped, but im tryna make a general code rn
    for selection in range(len(selected_permutations)):
        selectedIndex = selected_permutations[selection]
        selected = parents_permutations[selectedIndex]
        #this bit is to take half from one parent and half from another. 
        #this allows me to create the next generation as an improvement from the current 
        # one(taking values from the best chromosomes, and inputting them as all chromos in the next gen)
        halfway = int(newPopulation.shape[1]/2)
        newPopulation[combIndex + selection][0:halfway] = parents[selected[0]][0:halfway]
        newPopulation[combIndex + selection][halfway:] = parents[selected[1]][halfway:]
    return newPopulation

def mutation(population, pctgToBeChanged = 0.1, numParents = 3):
    for index in range(numParents, population.shape[0]):
        #this takes random indices in the entire chromosone(0.1 of total number) and changes them to a random value. 
        #This is needed to mutate the values, otherwise there will be no change in the chromosome
        rand_inx = np.uint32(np.random.random(np.uint32(pctgToBeChanged/100 * population.shape[1])) * population.shape[1])
        newVals = np.uint8(np.random.random(rand_inx.shape[0]) * 256)
        population[index, rand_inx] = newVals
    return population

img_arr = cv2.imread("/Users/soumilrathi/Desktop/eiffel.jpg")
img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
img_rgb = makeImageSmall(img_rgb)
chromo = img2chromo(img_rgb)
population = initialPopulation(img_rgb.shape)
for iteration in range(50000):
    qualities = totalFitness(chromo, population)
    best_quality = np.max(qualities)
    print("Quality:", round(best_quality, 2), "| Iteration:", iteration+1)
    maxQuality = best_quality
        
    population = crossover(getParents(population, qualities, 3), img_rgb.shape, 8)
    population = mutation(population)

img_final = chromo2img(population, img_rgb.shape)
plt.imshow(img_final)
plt.show()
