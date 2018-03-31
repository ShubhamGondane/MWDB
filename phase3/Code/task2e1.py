
import pandas as pd
import numpy as np
from sktensor import dtensor, cp , ktensor, sptensor
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from itertools import chain
from collections import defaultdict
import sys
from scipy import spatial
from scipy import linalg
import operator

def relevance_feedback(out):
    ### CALCULATE RELEVANCE FEEDBACK ###
    out = out.values.tolist()
    rel = []
    irrel = []
    genre_df = pd.read_csv("phase3_dataset/mlmovies.csv") 
    genre_df = genre_df[:len(genre_df)/3]
    movie_list = list(set(genre_df['movieid']))
    #print(len(movie_list))
    genres = set()
    for row in range(len(genre_df)):
        g = genre_df.iloc[row]['genres']
        g = set(g.split("|"))
        genres = genres.union(g)
    ### FIND ALL GENRES ###
    genre_dict = {}
    i = 0
    for g in genres:
        genre_dict[g] = i
        i += 1
    #print(genre_dict)
    for i in range(5):
        r = input("Is movie " + str(int(out[i][0])) + " relevant to you?  Enter 1 for RELEVANT, 0 for Don't care, -1 for IRRELEVANT:   ")
        rel.append([int(out[i][0]),r])
        if(r == -1):
            irrel.append(out[i][0])
    
    rel = pd.DataFrame(rel,columns = ['movieid','rel'])

    rel_count = {1:0, -1:0, 0:0}
    for row in range(len(rel)):
        for k in rel_count.keys():
            if(k == int(rel.iloc[row]['rel'])):
                rel_count[k] += 1
    #print(rel_count)
    rel = rel.merge(genre_df, on = 'movieid', how = 'inner')
    #print(rel)
    genres_rel = set()
    for row in range(len(rel)):
        g = rel.iloc[row]['genres']
        g = set(g.split("|"))
        genres_rel = genres_rel.union(g)
    genres_rel = list(genres_rel)

    gen = {}
    
    for g in genres_rel:
        for k in genre_dict.keys():
            if (g == k):
                gen[k] = [genre_dict[k],0,0,0]
                
    #print(gen)

    index = []
    
    for i in range(len(gen)):
        index.append(i)

    rel_feed = pd.DataFrame(columns = ['genre','ri','ni','R','N','positive_weight','negative_weight'], index = index)
    for row in range(len(rel)):
        for k in gen.keys():
            if (k in rel.iloc[row]['genres']):
                l = gen[k]
                l[2]+=1
                if(rel.iloc[row]['rel'] == 1):
                    l[1]+=1
                    gen[k] = [l[0],l[1],l[2],l[3]]
                   
                if(rel.iloc[row]['rel'] == -1):
                    l[3]+=1
                    gen[k] = [l[0],l[1],l[2],l[3]]
                

    gen1 = list(gen.values())
    
    for row in range(len(gen1)):           
        rel_feed.loc[row]['genre'] = gen1[row][0]
        rel_feed.loc[row]['ri'] = [gen1[row][1],gen1[row][3]]
        rel_feed.loc[row]['ni'] = gen1[row][2]
        rel_feed.loc[row]['N'] = 5
        rel_feed.loc[row]['R'] = [rel_count[1],rel_count[-1]]
    rel_feed.fillna(0)
    
    rel_feed = rel_feed.values.tolist()

    for i in range(len(rel_feed)):
        rel_feed[i][5] = np.log(((rel_feed[i][1][0] + 0.5)/(rel_feed[i][3][0] - rel_feed[i][1][0] + 1)) / ((rel_feed[i][2] - rel_feed[i][1][0] + 0.5)/ (1 + rel_feed[i][4] - rel_feed[i][3][0] - rel_feed[i][2] + rel_feed[i][1][0])))
        rel_feed[i][6] = np.log(((rel_feed[i][1][1] + 0.5)/(rel_feed[i][3][1] - rel_feed[i][1][1] + 1)) / ((rel_feed[i][2] - rel_feed[i][1][1] + 0.5)/ (1 + rel_feed[i][4] - rel_feed[i][3][1] - rel_feed[i][2] + rel_feed[i][1][1])))
    #print(rel_feed)
    result = []
    for r in rel_feed:
        l= []
        l.append(r[0])
        l.append(r[5])
        l.append(r[6])
        result.append(l)

    return result,irrel

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

tensor,m, user_movies = arr,movie_list,movie_timestamp
irrel = []

while(True):
    m = movie_list
    #user_movies = list(user_movies.keys())
    m, user_movies = movie_list,movie_timestamp
    #user_movies = list(user_movies.keys())
    T = dtensor(tensor) # SCIKIT TENSOR REPRESENTATION
    seeds = movie_timestamp
### CP-DECOMPOSITION ###
    P, fit, itr, exectimes = cp.als(T, 5, init = 'nvecs')

### FACTOR MATRICES GENERATED ###
    Factor1 = P.U[0]
    DP = ktensor.totensor(P)
    UF = dtensor.unfold(DP,0)

    U, s, V = linalg.svd(UF) 
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

    movie_pagerank = pd.DataFrame(columns = ['movieid','pagerank'])
    m2 = pd.read_csv("Tensor_dataset/mlmovies.csv")
    movie_pagerank['movieid'] = m
    movie_pagerank['pagerank'] = pagerank
    movie_pagerank = movie_pagerank.sort_values(by='pagerank', ascending=False)
    

    top_movies = movie_pagerank[(-movie_pagerank.movieid.isin(seeds_weight.keys()))]
    top_movies = top_movies.merge(m2, on = 'movieid', how = 'inner')
    top_movies = top_movies.drop(['year'], axis = 1)
    top_movies = top_movies.merge(mltags, on='movieid', how = 'inner')
    top_movies = top_movies.drop(['userid'], axis = 1)
    top_movies = top_movies[~top_movies['movieid'].isin(irrel)]
    top_movies = top_movies.groupby(['movieid','pagerank','moviename','genres']).agg(lambda x: set(x))
    top_movies.reset_index(inplace = True)
    top_movies.set_index('movieid', inplace=True)
    top_movies = top_movies.sort_values(by = 'pagerank', ascending = False)
    print(top_movies.head(n=5))
    

    rel_feedback,irrel = relevance_feedback(top_movies.head(n=5).reset_index())
    #print(rel_feedback)


    for r in rel_feedback:
        for i in range(mat[:,:,genre_list.index(r[0])].shape[0]):
            for j in range(mat[:,:,genre_list.index(r[0])].shape[1]):
                if (mat[i,j,genre_list.index(r[0])] != 0):
                    mat[i,j,genre_list.index(r[0])] += r[1]
                    mat[i,j,genre_list.index(r[0])] -= r[2]
    
    #print(np.unique(mat, return_counts = True))
    tensor = mat
    cont = input("Want to continue? 1 for YES, 0 for no     ")
    if(cont == 0):
        break




