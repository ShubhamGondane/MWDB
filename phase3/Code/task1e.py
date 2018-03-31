import pandas as pd
import numpy as np
#import tensorly as tl
from sktensor import dtensor, cp , ktensor, sptensor
from sklearn.cluster import KMeans
from scipy import linalg
import sys
import operator

def load():
    user = sys.argv[1]
    ### LOAD DATA FOR TENSOR ###
    mlratings = pd.read_csv("Tensor_dataset/mlratings.csv")
    
    mltags = pd.read_csv("Tensor_dataset/mltags.csv")
    #mltags = mltags[:len(mltags)/10]
    genre_df = pd.read_csv("Tensor_dataset/mlmovies.csv") 
    #genre_df = genre_df[:len(genre_df)/3]
    movie_timestamp = {}
    temp1 = mlratings[mlratings.userid == int(user)]
    temp2 = mltags[mltags.userid == int(user)]
   

    temp1 = temp1.drop(['userid','imdbid','rating'],axis = 1)
    temp1 = temp1.values.tolist()

    temp2 = temp2.drop(['userid','tagid'],axis = 1)
    temp2 = temp2.values.tolist()

    for movie in temp1:
        movie_timestamp[movie[0]] = np.array(movie[1]).tolist()
    for movie in temp2:
        movie_timestamp[movie[0]] = np.array(movie[1]).tolist()



    #rating_list = list(set(mlratings['rating']))
    movie_list = list(set(genre_df['movieid']))
    tag_list = list(set(mltags['tagid']))

    genres = set()
    for row in range(len(genre_df)):
        g = genre_df.iloc[row]['genres']
        g = set(g.split("|"))
        genres = genres.union(g)

    genre_dict = {}
    i = 0
    for g in genres:
        genre_dict[g] = i
        i += 1
    #print(genre_dict)
    genre_list = []
    for k,v in genre_dict.items():
        genre_list.append(v)
    genre_list.sort()
    
    #print(genre_list)
    ### FIND MOVIE - GENRE PAIRS ###
    movie_genre = []
    
    for row in range(len(genre_df)):
        for key,value in genre_dict.items():
            l1 = []
            if (key in genre_df.iloc[row]['genres']):
                l1.append(genre_df.iloc[row]['movieid'])
                l1.append(value)
                movie_genre.append(l1)
              
    movie_genre = pd.DataFrame(movie_genre,columns = ['movieid','genre'])
    mltags = mltags.drop(['timestamp'], axis = 1)

    triple = pd.merge(movie_genre, mltags, on = 'movieid', how = 'inner') # VALID TRIPLES
    triple['value'] = 1
    #print(triple)
    #triple = triple.drop(['userid','Avg_rating'], axis = 1)
    triple = triple.values.tolist()
    
    

    movies = np.array(movie_list)
    tags = np.array(tag_list)
    genres = np.array(genre_list)
    m = len(movies)
    t = len(tags)
    g = len(genres)
    mat = np.zeros((len(movies), len(tags), len(genres))) # 3D ARRAY
    
    #print(mat.shape)
    index_movies = []
    index_tags = []
    index_genres = []

    for m in triple:
        index_movies.append(movie_list.index(m[0]))
        index_genres.append(genre_list.index(m[1]))
        index_tags.append(tag_list.index(m[3]))
    
    for i in range(len(index_tags)):
        mat[index_movies[i]][index_tags[i]][index_genres[i]] = 1
    arr = mat
    return(arr,movie_list,movie_timestamp)


tensor,m,seeds = load()

T = dtensor(tensor) # SCIKIT TENSOR REPRESENTATION

### CP-DECOMPOSITION ###
P, fit, itr, exectimes = cp.als(T, 5, init = 'random')

### FACTOR MATRICES GENERATED ###
Factor1 = P.U[0]
Factor2 = P.U[1]
Factor3 = P.U[2]

UF = ktensor.totensor(P)
UF1 = dtensor.unfold(UF,0)
#print(UF1.shape)

U, s, V = linalg.svd(UF1) 
U = U[:,:10]
#print(U.shape)
D_T = np.transpose(U)
movie_movie = np.dot(U, D_T)


seeds = sorted(seeds.items(), key=operator.itemgetter(1), reverse=True)
seeds_weight = {}
count = len(seeds)
for seed in seeds:
    seeds_weight[seed[0]] = count
    count -= 1

teleport = np.zeros(shape=(len(m), 1), dtype=float)
for seed in seeds_weight.keys():
    if(seed in m):
        teleport[m.index(seed)] = seeds_weight[seed]
    else:
        continue

alpha = 0.85
err = 0.001

        # Column normalize movie-movie matrix.
        # This matrix is transition matrix
movie_movie_norm = movie_movie / movie_movie.sum(axis=0)
np.warnings.filterwarnings('ignore')
movie_movie_norm = np.nan_to_num(movie_movie_norm)
size = movie_movie_norm.shape[0]
#print(size)
t = np.array(teleport)
pagerank = np.ones(size)
prev = np.zeros(size)

        # Calculate pagerank
while np.sum(np.abs(pagerank - prev)) > err:
    prev = pagerank
    pagerank = ((1 - alpha) * np.dot(movie_movie_norm, pagerank)) + (alpha * t)
    np.warnings.filterwarnings('ignore')
    
mltags = pd.read_csv("Tensor_dataset/mltags.csv")
#mltags = mltags[:len(mltags)/10]
movie_pagerank = pd.DataFrame(columns = ['movieid','pagerank'])
m2 = pd.read_csv("Tensor_dataset/mlmovies.csv")
movie_pagerank['movieid'] = m
movie_pagerank['pagerank'] = pagerank
movie_pagerank = movie_pagerank.sort_values(by='pagerank', ascending=False)

### RECOMMEND MOVIES ###
top_movies = movie_pagerank[(-movie_pagerank.movieid.isin(seeds_weight.keys()))]
top_movies = top_movies.merge(m2, on = 'movieid', how = 'inner')
top_movies = top_movies.merge(mltags, on='movieid', how = 'inner')
top_movies = top_movies.drop(['timestamp','userid','year'], axis = 1)
top_movies = top_movies.groupby(['movieid','pagerank','moviename','genres']).agg(lambda x: set(x))
top_movies.reset_index(inplace = True)
top_movies.set_index('movieid', inplace=True)
top_movies = top_movies.sort_values(by = 'pagerank', ascending = False)
print(top_movies.head(n=5))


