
import pandas as pd
import numpy as np
from sktensor import dtensor, cp , ktensor, sptensor
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from itertools import chain
from collections import defaultdict
import sys
from scipy import spatial

# Create a user-movie-tag OR user-movie-genre tensor then using cp-decomposition we have to compare the user factor matrix's row vectors and find similar 
# vectors to the input user.

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

tensor,m, user_movies = load() 
user_movies = list(user_movies.keys())

T = dtensor(tensor) # SCIKIT TENSOR REPRESENTATION
mltags = pd.read_csv("Tensor_dataset/mltags.csv")
#mltags = mltags[:len(mltags)/10]
### CP-DECOMPOSITION ###
P, fit, itr, exectimes = cp.als(T, 5, init = 'random')

### FACTOR MATRICES GENERATED ###
Factor1 = P.U[0]
Factor2 = P.U[1]
Factor3 = P.U[2]

movies = pd.DataFrame(m,columns = ['movieid'])

for i in range(5):
    movies['ls'+ str(i+1)] = Factor1[:,i]

l1 = []
l2 = {}
for m in user_movies:
    if(any(movies.movieid == int(m))):
        l = movies[movies.movieid == int(m)].values.tolist()
        l2[l[0][0]] = l[0][1:6]
um = l2
movies = movies.set_index('movieid').T.to_dict('list')
reco = []
for k1,v1 in movies.items():
    for k2,v2 in um.items():
        key = [k2,k1]
        key.append(1-spatial.distance.cosine(np.asarray(v2), np.asarray(v1)))
        np.warnings.filterwarnings('ignore')
        reco.append(key)
        
recommend = pd.DataFrame(reco,columns = ['user','movieid','cosine-similarity'])
recommend = recommend.sort_values(by = 'cosine-similarity', ascending = False)

#recommend = recommend[recommend.movieid != 0]
top_movies = recommend[~recommend.movieid.isin(user_movies)]

### RECOMMEND MOVIES ###
m2 = pd.read_csv("Tensor_dataset/mlmovies.csv")
top_movies = top_movies.merge(m2, on = 'movieid', how = 'inner')
top_movies = top_movies.drop(['year','user'], axis = 1)
top_movies = top_movies.drop_duplicates('movieid',keep = 'first')
top_movies = top_movies[~top_movies['cosine-similarity'].isin(['inf'])]
top_movies = top_movies.merge(mltags, on='movieid', how = 'inner')
top_movies = top_movies.drop(['timestamp','userid'], axis = 1)
top_movies = top_movies.groupby(['movieid','cosine-similarity','moviename','genres']).agg(lambda x: set(x))
top_movies.reset_index(inplace = True)
top_movies.set_index('movieid', inplace=True)
top_movies = top_movies.sort_values(by = 'cosine-similarity', ascending = False)
print(top_movies.head(n=5))

