
import pandas as pd
import math
import sys
import numpy as np
import datetime as dt
from operator import add
# import print_genre_vector_rtr as pgv
# import print_genre_vector2_rtr as pgv2
import lda
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ldaLatentSemantics(iternum):
    gentag=pd.read_csv("E:/MWDB_fall2017/task3_phase3/phase3_dataset/genome-tags.csv").set_index("tagId")
    movienamedf=pd.read_csv("E:/MWDB_fall2017/task3_phase3/phase3_dataset/mlmovies.csv")
    movienamedf=movienamedf[["movieid","moviename"]].set_index("movieid")

    dataFrame=pd.read_csv("E:/MWDB_fall2017/task3_phase3/phase3_dataset/mltags.csv")
    dataFrame=dataFrame[["movieid","tagid"]]
    dataFrame.insert(2,'D',1)

    dataFrame=dataFrame.pivot_table(values='D',index='movieid', columns='tagid',aggfunc=np.sum , fill_value=0)
    #dataFrame.to_csv("/Users/RTR/Desktop/Mypro/github/data/movie-recommendation/movietags.csv")

    #dataFrame=pd.read_csv("/Users/RTR/Desktop/Mypro/github/data/movie-recommendation/movietags.csv").set_index("movieid")

    dataFrame = dataFrame.dropna(axis=1, how='all')
    dataFrame = dataFrame.dropna(axis=0, how='all')
    dataFrame = dataFrame.fillna(0)

    vocab = dataFrame.columns.copy().tolist()
    vl=[]
    for i in vocab:
        vl.append(gentag.loc[int(i)]["tag"])
    vocab=vl.copy()
    vocab = tuple(vocab)
    ind_g = dataFrame.index.copy()
    



    doctm = dataFrame.as_matrix()
    doctm = doctm.astype(int)
    model = lda.LDA(n_topics=500, n_iter=iternum, random_state=1)
    model.fit(doctm)
    topic_word = model.topic_word_
    n_top_words = 10
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(
        topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ','.join(topic_words)))
    c = model.transform(doctm)
    c = pd.DataFrame(c, index=ind_g)
    c.to_csv("E:/MWDB_fall2017/task3_phase3/t3/movietagsldapac500_1500.csv")
    return c
