import random
#Find x, y, z which satisfy this equation(6*x**3+ 9*y**2 + (90*z)) when it equals 25

def calcValue(x, y, z):
    return 6*x**3+ 9*y**2 + (90*z) - 25 #subtracted 25 so that we can judge algorithm based on closeness to 0
def calcActual(x, y, z):
    return 6*x**3+ 9*y**2 + (90*z)

def fitnessFunction(x, y, z):
    ans = calcValue(x, y, z)
    if(ans == 0):
        return 99999999
    else:
        return abs(1/ans)
    #The larger this return value, the better the function


#generate solutions
solutions = []
for s in range(1000):
    solutions.append((random.uniform(0,10000), random.uniform(0,10000), random.uniform(0, 10000)) )

#For all solutions, rank them.

for i in range(10000):
    rankedSolutions = []
    for s in solutions:
        rankedSolutions.append( (fitnessFunction(s[0], s[1], s[2]), s)  )
    rankedSolutions.sort()
    rankedSolutions.reverse()
    
    print(f"==Gen {i} best solution ==")
    print(rankedSolutions[0])

    if(rankedSolutions[0][0] > 2000000):
        print(calcActual(rankedSolutions[0][1][0],rankedSolutions[0][1][1],rankedSolutions[0][1][2] ))
        break
    #Take the best few solutions and combine them into like BEST solution
    bestSolutions =  rankedSolutions[:100]
    elements = []
    newGen = []
    for s in bestSolutions:
        #print(s)
        newGen.append(s[1])
        elements.append(s[1])
        elements.append(s[1])
        elements.append(s[1])
    


    #These elements(parameters x, y, z) also have to be mutated
    for r in range(1000):
        e1  = random.choice(elements)[0] * random.uniform(0.99, 1.01)
        e2 = random.choice(elements)[1] * random.uniform(0.99, 1.01)
        e3 = random.choice(elements)[2] * random.uniform(0.99, 1.01)
        newGen.append( (e1, e2, e3) )
    solutions = newGen

