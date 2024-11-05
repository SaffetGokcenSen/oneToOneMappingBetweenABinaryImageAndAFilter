# gradient descent optimization with weighing and Nesterov momentum

import numpy as np 
from itertools import product 

# possible values of the pixels
g = [-1, 1] 

# number of pixels of the filter 
gridSize = 6
# all the permutations of -1 and 1 in the filter of size gridSize. the number 
# of permutations is 2 to the power of gridSize.
permutations = np.array(list(product(g, repeat=gridSize))) 

# the difference between a permutation and the other permutations are 
# computed.each set of differences form a matrix and each matrix is put into a 
# list.      
theSize = 0 
diffList = []
for i in range(permutations.shape[0]-1):
    theVec = permutations[i, :] 
    theDiff = theVec-permutations[i+1:, :] 
    theSize = theSize + theDiff.shape[0]
    diffList.append(theDiff)

# the matrices in the list are concatenated and collected in a matrix.
theDiffMatrix = np.ones((theSize, gridSize))
startRow = 0
for i in range(len(diffList)): 
    currentMatrix = diffList[i]
    stopRow = startRow + currentMatrix.shape[0]
    theDiffMatrix[startRow:stopRow, :] = currentMatrix 
    startRow = startRow + currentMatrix.shape[0] 

# it is checked if the concatenation has been correctly made.
startRow = 0
for i in range(len(diffList)): 
    currentMatrix = diffList[i]
    stopRow = startRow + currentMatrix.shape[0]
    array1 = theDiffMatrix[startRow:stopRow, :] 
    if not np.array_equal(currentMatrix, array1): 
        print(i)
        break
    startRow = startRow + currentMatrix.shape[0] 

# the difference matrix can have duplicate rows. they are removed.
reducedDiffMatrix, indices = np.unique(theDiffMatrix, axis=0, return_index=True) 

# the optimization of the filter coefficients must be made in such a way that 
# there is a one-to-one correspondence between the pixel grid values and the 
# inner product of the filter and the pixel grid. How can this be done? It is 
# going to be done using the gradient descent algorithm. At each step of the 
# training, the inner product of the filter with each row of the difference 
# matrix is done. The products which are in a certain neighborhood of zero 
# are determined. The rows corresponding to these products give the directions 
# of the update for the filter. 

# First, a random filter is created. The filter is updated until one-to-one 
# correspondence is satisfied. let a random filter be generated 
rng = np.random.default_rng(seed=4) 
randFloatVec = rng.random(gridSize) 
print("initial random filter:")
print(randFloatVec)
# the inner product of the filter with each row of the reduced difference 
# matrix is performed.
innerProducts = np.sum(reducedDiffMatrix * randFloatVec, axis=1) 
# the inner products whose magnitudes are within a particular neighborhood 
# of 0 are determined. Let the radius of the neightborhood be set.
neighRadius = 0.2
# the magnitudes of all the inner products 
allMags = np.abs(innerProducts)
# the inner products in the neighborhood are found.
directionIndices  = np.argwhere(allMags <= neighRadius).flatten() 
# number of iterations done so far in the optimization.
iterNum = 0 
# the list holding numbers of unsatisfied conditions for the one-to-one mapping. 
problematicRowNumList = [] 
# add the current number of unsatisfied conditions to the list.
problematicRowNumList.append(directionIndices.shape[0]) 
# the list holding the evolution of the filter.
floatVecList = [] 
# add the current filter to the list.
floatVecList.append(randFloatVec)
stepSize = 0.01 
# previous gradients contribution 
b = 0.0 
# the momentum constant 
mu = 0.9
while (directionIndices.shape[0] != 0) & (iterNum <= 2000000): 
    # the look-ahead vector is calculated to implement Nesterov correction 
    randFloatVecLookAhead = randFloatVec + mu * b

    # the gradient at the look-ahead vector is calculated
    # the inner product of the look-ahead filter with each row of the reduced 
    # difference matrix is performed.
    innerProductsLookAhead = np.sum(
            reducedDiffMatrix * randFloatVecLookAhead, axis=1) 
    # the magnitudes of all the look-ahead inner products 
    allMagsLookAhead = np.abs(innerProductsLookAhead)
    # the look-ahead inner products in the neighborhood are found.
    directionIndicesLookAhead  = np.argwhere(
            allMagsLookAhead <= neighRadius).flatten()
    # the corresponding rows of the reduced difference matrix are got for the 
    # lookahead filter. 
    unnormalizedGradDirectionsLookAhead = reducedDiffMatrix[
            directionIndicesLookAhead]
    norms = np.linalg.norm(unnormalizedGradDirectionsLookAhead)
    gradDirectionsLookAhead = unnormalizedGradDirectionsLookAhead / norms
    # find the norms of the look-ahead inner products which do not satisfy 
    # one-to-one mapping conditions.
    smallMagsLookAhead = allMagsLookAhead[directionIndicesLookAhead] 
    # weights to be applied to the problematic directions are calculated for the 
    # look-ahead filter.
    smallMagsSumLookAhead = np.sum(smallMagsLookAhead)
    smallMagsLookAhead = np.reshape(
            allMagsLookAhead[directionIndicesLookAhead], (-1, 1)) 
    # the weights are applied to the problematic directions for the case of
    # look-ahead filter.
    gradDirectionsLookAhead = gradDirectionsLookAhead / smallMagsLookAhead
    # the unnormalized direction is calculated for the case of look-ahead filter.
    unnormalizedDirectionLookAhead = np.sum(gradDirectionsLookAhead, axis=0) 
    # the normalized direction is calculated for the case of look-ahead filter. 
    theNorm = np.linalg.norm(unnormalizedDirectionLookAhead) 
    if theNorm != 0.0:
        theDirectionLookAhead = unnormalizedDirectionLookAhead / theNorm
    b = mu * b + theDirectionLookAhead  
    randFloatVec = randFloatVecLookAhead + stepSize * theDirectionLookAhead

    # the inner product of the filter with each row of the reduced difference 
    # matrix is performed.
    innerProducts = np.sum(reducedDiffMatrix * randFloatVec, axis=1) 
    # the magnitudes of all the inner products 
    allMags = np.abs(innerProducts)
    # the inner products in the neighborhood are found.
    directionIndices  = np.argwhere(allMags <= neighRadius).flatten()
    # add the current number of unsatisfied conditions to the list.
    problematicRowNumList.append(directionIndices.shape[0])
    # add the current filter to the list.
    floatVecList.append(randFloatVec) 

    iterNum = iterNum + 1 

# the number of iterations the optimization has taken.
print("iteration number:") 
print(iterNum) 
# the final filter satisfying the one-to-one mapping condition.
print("final random filter:")
print(randFloatVec) 

import matplotlib.pyplot as plt 
fig, ax = plt.subplots() 
# the variation of the number of unsatisfied conditions with respect to the 
# iteration number
ax.plot(problematicRowNumList) 
ax.set_xlabel("number of iterations") 
ax.set_ylabel("number of unsatisfied conditions") 
ax.set_title("iteration number versus number of unsatisfied conditions")
plt.savefig("iterations_vs_unsatisfied_coditions4.png")

# the matrix holding all the filters produced during optimization is initialized. 
randFloatVecMatrix = np.zeros((len(floatVecList), gridSize)) 
# the matrix holding all the filters produced during optimization is populated. 
i = 0
for theVector in floatVecList: 
    randFloatVecMatrix[i, :] = theVector 
    i = i + 1 

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, sharey=True)
fig.suptitle("the evolution of the filter components")

# the evolution of the first component of the filter is plotted.
ax1.plot(randFloatVecMatrix[:, 0]) 
ax1.set_xlabel("iteration") 
ax1.set_ylabel("1st component") 

# the evolution of the second component of the filter is plotted.
ax2.plot(randFloatVecMatrix[:, 1]) 
ax2.set_xlabel("iteration") 
ax2.set_ylabel("2nd component") 

# the evolution of the third component of the filter is plotted.
ax3.plot(randFloatVecMatrix[:, 2])
ax3.set_xlabel("iteration")
ax3.set_ylabel("3rd component")

# the evolution of the fourth component of the filter is plotted.
ax4.plot(randFloatVecMatrix[:, 3])
ax4.set_xlabel("iteration")
ax4.set_ylabel("4th component")

# the evolution of the fifth component of the filter is plotted.
ax5.plot(randFloatVecMatrix[:, 4])
ax5.set_xlabel("iteration")
ax5.set_ylabel("5th component")

# the evolution of the sixth component of the filter is plotted.
ax6.plot(randFloatVecMatrix[:, 5])
ax6.set_xlabel("iteration")
ax6.set_ylabel("6th component")
plt.savefig("filter_evolution4.png")

finalInnerProducts = np.sum(permutations * randFloatVec, axis=1) 

# check if the distance between any two identities is smaller than or equal 
# to the set threshold
for i in range(finalInnerProducts.shape[0]):
    x = np.delete(finalInnerProducts, np.arange(0, (i+1))) 
    if ((np.abs(finalInnerProducts[i]-x) <= neighRadius).sum() > 0): 
        print("There is a problem!")
        break

print("final inner products:")
print(finalInnerProducts)
