#!/anaconda/envs/main/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 14:14:27 2017

@author: shubhamgondane
"""

import pandas as pd
import numpy as np
import sys 
import pprint
'''         Input and read datasets          '''

#actorid = input("Enter actorid\n")
actorid = sys.argv[1]
model = sys.argv[2]

movies= pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/movie-actor.csv")
movie_actor = movies[movies.actorid == int(actorid)]
movie_actor = pd.DataFrame(movie_actor,columns=['movieid','actorid','actor_movie_rank'])
movie_actor.reset_index()
movie_actor.set_index('movieid',inplace = True)
movie_actor.reset_index(level=0, inplace=True)

shape = movie_actor.shape
total_movies = shape[0]   # TOTAL MOVIES FOR CALCULATING IDF

mltag = pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/mltags.csv")

#######################################################################

'''         Actor rank weights         '''

movie_actor['actor_movie_rank'] = 1/movie_actor['actor_movie_rank']
total_ranks = movie_actor['actor_movie_rank'].sum()  #Using a weighting scheme
movie_actor['rank_weights'] = movie_actor['actor_movie_rank']/total_ranks

movie_actor_tag = mltag.merge(movie_actor, on ='movieid')


tag_rank_pair = movie_actor_tag[['tagid','rank_weights']].copy()
tag_rank_pair= tag_rank_pair.drop_duplicates()
movie_actor_tag = movie_actor_tag.sort_values(by='timestamp',ascending=False)  # For each movie and the tags contained in it I am sorting it in order of newest tag first

#######################################################################
'''       Time stamp Weights            '''

timerank = []
movie_tag_actor_rows = movie_actor_tag.shape
for i in range(1,movie_tag_actor_rows[0]+1):
    timerank.append(i)

timerank = np.divide(1,timerank)
time_rank_weights = timerank/(np.sum(timerank,dtype = np.float64))
movie_actor_tag['time_rank_weights']=timerank
#print(movie_actor_tag)

#######################################################################

'''      Term frequency of tags for each movie      '''
movie_tag_count = movie_actor_tag.groupby(['movieid']).size().reset_index(name = 'total_tag_count')  

from collections import Counter
tag_movie_pair = {k: g["tagid"].tolist() for k,g in movie_actor_tag.groupby(['movieid'])} 
random = {}
final = {}
tf_tag = {}
for key,values in tag_movie_pair.items():
    total_tags = len(values)
    c = Counter(values)
    a = {k: v / total_tags for k, v in c.items()}
    final[key] = {}
    for k,v in a.items():
        final[key][k] = v
    
# The above loop calculates the term frequency and tagid pairs for each movie seperately.
#######################################################################
        
final_df=pd.DataFrame.from_dict({(i,j): final[i][j]   #STORE ALL THE TERM FREQ CALCULATED IN A DATAFRAME
                           for i in final.keys() 
                           for j in final[i].keys()},
                       orient='index') 
final_df.reset_index(inplace = True)
#print(final_df)
final_df.columns = ['movie_tag','tf']
final_df[['movieid','tagid']] = final_df['movie_tag'].apply(pd.Series)
#print(final_df)
final_df = final_df.drop('movie_tag',1)

ab = movie_actor_tag.merge(final_df,on='tagid')
#print(ab)

'''        IDF CALCULATION          '''
### FOR EACH TAG --> FIND THE NUMBER OF MOVIES IN WHICH A TAG IS PRESENT 
tag_in_movie = {k: g["movieid"].tolist() for k,g in movie_actor_tag.groupby(['tagid'])} # gives me list of movies in which a particular tag appears
count_tag_in_movie = {}
for key,values in tag_in_movie.items():
    count_tag_in_movie[key] = len(values)
    
count_tag_in_movies = pd.Series(count_tag_in_movie, name = 'No_of_movies')
count_tag_in_movies.reset_index()
count_tag_in_movies.index.names = ['tagid']
count_tag_in_movies = pd.DataFrame(count_tag_in_movies)
count_tag_in_movies.reset_index(inplace = True)

#######################################################################

'''         AVG OUT TIME WEIGHTS FOR TAGS APPEARING MANY TIMES          '''
#print(movie_actor_tag[['tagid','time_rank_weights']])
tag_timestamp = {k: g["time_rank_weights"].tolist() for k,g in movie_actor_tag.groupby(['movieid','tagid'])} # to give uniform timestamp weights to each tag 
#print(tag_timestamp)
count_timestamps= {}
for key,values in tag_timestamp.items():
    count = len(values)
    ts = list(values)
    #count_timestamps[key] = np.sum(ts)/count
    count_timestamps[key] = np.sum(ts)  # sum 

  

count_timestamps = pd.Series(count_timestamps,name = 'final_time_weights')
count_timestamps.reset_index()
#print(count_timestamps)
count_timestamps.index.names = ['movieid','tagid']
count_timestamps = pd.DataFrame(count_timestamps)
count_timestamps.reset_index(inplace = True)
#print(count_timestamps)

#######################################################################

'''            COMBINE ALL THE WEIGHTS                '''

tfdf = final_df.merge(count_timestamps,on='tagid') 
tfdf = tfdf.merge(movie_actor_tag,on='tagid')
movie_actor_tag = movie_actor_tag.groupby(['movieid','tagid','timestamp','time_rank_weights','rank_weights']).size().reset_index(name='count')   #calculate the number of times a tag appears in movie and 

 # TF and TIME WEIGHTS
idf = pd.merge(movie_actor_tag,count_tag_in_movies, on = 'tagid')
idf['idf'] = np.log10(total_movies/idf['No_of_movies'])  # ACTUAL IDF CALCULATION
idf = idf.merge(count_timestamps, on = 'tagid')
idf = idf.drop('time_rank_weights',1)

tfdf = tfdf.groupby(['tagid'])['tf'].sum()
tfdf = pd.DataFrame(tfdf)
tfdf.reset_index(inplace = True)
tfdf.columns = ['tagid','tf']
tfdf = tfdf.merge(count_timestamps,on='tagid')
tfdf = tfdf.merge(tag_rank_pair,on='tagid')

########################################################################

'''              Tags and Weights pair              '''

genome_tags = pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/genome-tags.csv")
tfdf['final_weights'] = tfdf['tf']+tfdf['final_time_weights']+tfdf['rank_weights']

tfidf = tfdf

tfidf['idf'] = idf['idf'] ### TFIDF AND TIME WEIGHTS

tfdf = tfdf.merge(genome_tags,on='tagid')
tfdf = tfdf.sort_values(by='final_weights',ascending = False)

tfidf['final_weights'] = ((tfidf['tf']+tfidf['final_time_weights'])*tfidf['idf']+tfidf['rank_weights'])
tfidf = tfidf.merge(genome_tags,on='tagid')
tfidf = tfidf.sort_values(by='final_weights',ascending = False)
########################################################################

'''                   Output                        '''
if(model == 'TF'):
    pprint.pprint(list(zip(tfdf.tag,tfdf.final_weights)))
elif(model == 'TF-IDF'):
    pprint.pprint(list(zip(tfidf.tag,tfidf.final_weights)))
else:
    print("Wrong model entered!")
