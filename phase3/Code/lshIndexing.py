import pandas as pd
import math
import sys
import numpy as np
import datetime as dt
import ldaMovies
import matplotlib.pyplot as plt
from operator import add

def getLSHIndexTool(numberOfHashesPerLayer,numberOfLayers,width,dim):
    
    # getting the movie vector represented using 500 latent semantics
    dataPoints= ldaMovies.ldaLatentSemantics(50)
    mainDict={}
    count = 0
    
    # Generating Random Vectors for all the hashes
    nk = [[] for point in range(numberOfHashesPerLayer)]
    hashfunctions=[nk.copy() for point in range(numberOfLayers)]

    for point in range(numberOfLayers):
        for j in range(numberOfHashesPerLayer):
            randomVector = np.random.normal(0,1,dim).tolist()
            randomVector.append(np.random.uniform(0,width))
            hashfunctions[point][j]=randomVector.copy()
    hashfunctions=np.array(hashfunctions)
    
    # Constructing the index structures for all the layers
    for layerOfHashes in hashfunctions:
        indexStructureOfLayer={}
        cou_m=0
        for point in dataPoints.index:
            cou_m+=1

            movieId = point
            pointVector = dataPoints.loc[point].tolist()
            movieHashCodeOfLayer = getHashCodeOfPointForALayer(layerOfHashes, pointVector, width)
            hashCode =''.join(str(each) for each in movieHashCodeOfLayer)
            
            if hashCode in indexStructureOfLayer.keys():
                indexStructureOfLayer[hashCode].append(movieId)
            else:
                indexStructureOfLayer[hashCode]=[movieHashCodeOfLayer,movieId]

        mainDict[count]=indexStructureOfLayer.copy()
        count = count + 1

    return(mainDict, hashfunctions, dataPoints)



def getHashCodeOfPointForALayer(layerOfHashes, pointVector, width):
    movieHashCodeOfLayer = []
    for hash in layerOfHashes:
        randomProjectionR = hash[:-1].copy()
        r=hash[-1].copy()
        value = math.floor(randomProjectionR.dot(pointVector) +r)/width
        movieHashCodeOfLayer.append(value)
    return movieHashCodeOfLayer



def evaluateHammingDist(hashCode1, hashCode2):
    dist=0
    for index in range(len(hashCode1)):
        if hashCode1[index]!=hashCode2[index]:
            dist+=1
    return dist


# Hashing the given query on the lshIndex tool and retrieving the most similar movies
def indexingAQueryToFindSimilarMovies(mainDict,hashfunctions,dataPoints,queryVector,numberOfMoviesSimilarToOutput, width, queryMovieId):
    hashCodes_OfAllLayers=[]
    for layerOfHashes in hashfunctions:
        movieHashCodeOfLayer = getHashCodeOfPointForALayer(layerOfHashes, queryVector, width)
        hashCodes_OfAllLayers.append(movieHashCodeOfLayer)
        
    hammingDistance=-1
    queryDict={}

    while(len(queryDict) < numberOfMoviesSimilarToOutput):
        hammingDistance+=1
        movieListSimilar=[]
        indexTable=0
        for hashCodes in hashCodes_OfAllLayers:
            for index in mainDict[indexTable].keys():
                x=mainDict[indexTable][index][0]
                if evaluateHammingDist(hashCodes,x)<=hammingDistance:
                    movieListSimilar.extend(mainDict[indexTable][index][1:])
            indexTable = indexTable + 1
        queryDict = {layerOfHashes:movieListSimilar.count(layerOfHashes) for layerOfHashes in movieListSimilar}
#            print("hamming distance used = ",hammingDistance)
    
    queryDictUpdated = {}
    queryDict = {movieid:movieListSimilar.count(movieid) for movieid in movieListSimilar}
    
    for movieId in queryDict.keys():
        if movieId != queryMovieId:
            queryDictUpdated[movieId]= queryDict[movieId]

    print("Unique movies which are similar to given movie",len(queryDictUpdated.keys()))
    print("All movies similar to the given movie ",sum(queryDictUpdated.values()))
    
    # reversing the items in dictionary based on the count of the movieId
    queryDictUpdated=sorted(queryDictUpdated.items(), key=lambda x:x[1],reverse=True)

    dotProductDict = {}
    for movieId in queryDictUpdated:
        vector=dataPoints.loc[movieId[0]].tolist()
        vector=np.array(vector)
        dotProductDict[movieId[0]] = vector.dot(queryVector)
    
    nearestNeighbors=sorted(dotProductDict.items(), key=lambda x:x[1],reverse=True)
    nearestNeighbors=nearestNeighbors[:numberOfMoviesSimilarToOutput]
    print(nearestNeighbors)

        # maj_rl=[]
        # for layerOfHashes in maj_dict:
        #     maj_rl.append(layerOfHashes[0])
    result = []
    for movieId in nearestNeighbors:
        result.append(movieId[0])

    mlMoviesData = pd.read_csv("E:/MWDB_fall2017/task3_phase3/phase3_dataset/mlmovies.csv")
    mlMoviesData = mlMoviesData[["movieid","moviename"]].set_index("movieid")

    resultSimilarMovieName = []
    for layerOfHashes in result:
        resultSimilarMovieName.append(mlMoviesData.loc[layerOfHashes]["moviename"])
    print(resultSimilarMovieName)
    return(result, mlMoviesData)



def lshIndexing(numberOfHashesPerLayer, numberOfLayers, queryMovieId, numberOfMoviesSimilarToOutput):
    
    width = 1
    dimensions=500
    numberOfMoviesSimilarToOutput = int(numberOfMoviesSimilarToOutput)
    numberOfHashesPerLayer = int(numberOfHashesPerLayer)
    numberOfLayers = int(numberOfLayers)
    queryMovieId = int(queryMovieId)

    mainDict,hashfunctions,dataPoints = getLSHIndexTool(numberOfHashesPerLayer,numberOfLayers,width,dimensions)
    queryVector = dataPoints.loc[queryMovieId]
    results,mlMoviesData = indexingAQueryToFindSimilarMovies(mainDict,hashfunctions,dataPoints,queryVector,numberOfMoviesSimilarToOutput, width, queryMovieId)
    
    while(True):
        relevant_movies = []
        irrelevant_movies = []
        for result in results:
            print("Do you like this movie y or n", result, "  ", mlMoviesData.loc[result]["moviename"])
            feedback = input()
            curr_vector = dataPoints.loc[result].tolist()
            if feedback == "y":
                relevant_movies.append(curr_vector/np.linalg.norm(curr_vector,2))
            else:
                irrelevant_movies.append(curr_vector/np.linalg.norm(curr_vector,2))
        
        # Relevance Feedback
        part1 = [sum(x) for x in zip(*relevant_movies)]
        part2 = [sum(x) for x in zip(*irrelevant_movies)]
        queryVector = queryVector + [elem/len(relevant_movies) for elem in part1] - [elem/len(irrelevant_movies) for elem in part2]
        results,mlMoviesData = indexingAQueryToFindSimilarMovies(mainDict,hashfunctions,dataPoints,queryVector,numberOfMoviesSimilarToOutput, width, queryMovieId)
    
lshIndexing(numberOfHashesPerLayer=sys.argv[1],numberOfLayers=sys.argv[2],queryMovieId=sys.argv[3],numberOfMoviesSimilarToOutput=sys.argv[4])

