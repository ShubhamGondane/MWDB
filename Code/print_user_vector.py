#!/anaconda/envs/main/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 00:47:52 2017

@author: shubhamgondane
"""

import pandas as pd
import numpy as np
import sys
import pprint

'''            Input userid and  read datasets           '''

#userid = input("Enter Userid\n")
userid = sys.argv[1]
model = sys.argv[2]
movies= pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/mltags.csv")
movie_user = movies[movies.userid == int(userid)]
mlratings = pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/mltags.csv")
movie_rated = mlratings[mlratings.userid== int(userid)]

movie_rated = pd.DataFrame(movie_rated,columns=['movieid'])
#######################################################################

'''  User may have watched and rated a movie but didn't write a tag '''

movie_rated = movie_rated.merge(movies,on='movieid')
movie_rated = movie_rated[movie_rated.userid ==int(userid)]
movie_user = movie_user.merge(movie_rated,on='movieid')
movie_user = movie_user.drop('timestamp_y',1)
movie_user = movie_user.drop('userid_y',1)
movie_user = movie_user.drop('tagid_y',1)
movie_user.columns = ['userid','movieid','tagid','timestamp']
#print(movie_user)


movie_user = movie_user.sort_values(by='timestamp', ascending = False)
shape = movie_user.shape
total_movies = shape[0]   # TOTAL MOVIES FOR CALCULATING IDF

#######################################################################

'''                Time stamp weights               '''
movie_user_tag = movie_user  # For each movie and the tags contained in it I am sorting it in order of newest tag first

timerank = []
movie_tag_user_rows = movie_user_tag.shape
for i in range(1,movie_tag_user_rows[0]+1):
    timerank.append(i)

timerank = np.divide(1,timerank)
time_rank_weights = timerank/(np.sum(timerank,dtype = np.float64))
movie_user_tag['time_rank_weights']=timerank
#######################################################################

'''        Term frequency of tags for each movie           '''
movie_tag_count = movie_user_tag.groupby(['movieid']).size().reset_index(name = 'total_tag_count')  

from collections import Counter
tag_movie_pair = {k: g["tagid"].tolist() for k,g in movie_user_tag.groupby(['movieid'])} 
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
final_df.columns = ['movie_tag','tf']
final_df[['movie','tagid']] = final_df['movie_tag'].apply(pd.Series)
final_df = final_df.drop('movie_tag',1)

'''            IDF CALCULATION              ''' 
### FOR EACH TAG --> FIND THE NUMBER OF MOVIES IN WHICH A TAG IS PRESENT 
tag_in_movie = {k: g["movieid"].tolist() for k,g in movie_user_tag.groupby(['tagid'])} # gives me list of movies in which a particular tag appears
count_tag_in_movie = {}
for key,values in tag_in_movie.items():
    count_tag_in_movie[key] = len(values)
    
count_tag_in_movies = pd.Series(count_tag_in_movie, name = 'No_of_movies')
count_tag_in_movies.reset_index()
count_tag_in_movies.index.names = ['tagid']
count_tag_in_movies = pd.DataFrame(count_tag_in_movies)
count_tag_in_movies.reset_index(inplace = True)
#######################################################################

''' AVG OUT TIME WEIGHTS FOR TAGS APPEARING MANY TIMES '''
tag_timestamp = {k: g["time_rank_weights"].tolist() for k,g in movie_user_tag.groupby(['tagid'])} # to give uniform timestamp weights to each tag 
count_timestamps= {}
for key,values in tag_timestamp.items():
    count = len(values)
    ts = list(values)
    count_timestamps[key] = np.sum(ts)
    #count_timestamps[key] = np.sum(ts)/count
    
count_timestamps = pd.Series(count_timestamps,name = 'final_time_weights')
count_timestamps.reset_index()
count_timestamps.index.names = ['tagid']
count_timestamps = pd.DataFrame(count_timestamps)
count_timestamps.reset_index(inplace = True)

#######################################################################

'''          Combine all the weights            '''
movie_user_tag = movie_user_tag.groupby(['movieid','tagid','timestamp','time_rank_weights']).size().reset_index(name='count')   #calculate the number of times a tag appears in movie and 

tfdf = final_df.merge(count_timestamps,on='tagid')  # TF and TIME WEIGHTS
idf = pd.merge(movie_user_tag,count_tag_in_movies, on = 'tagid')
idf['idf'] = np.log10(total_movies/idf['No_of_movies'])  # ACTUAL IDF CALCULATION

idf = idf.merge(count_timestamps, on = 'tagid')
idf = idf.drop('time_rank_weights',1)

tfdf = tfdf.groupby(['tagid'])['tf'].sum()
tfdf = pd.DataFrame(tfdf)
tfdf.reset_index(inplace = True)
tfdf.columns = ['tagid','tf']
tfdf = tfdf.merge(count_timestamps,on='tagid')
#######################################################################

'''             Tag and weights pair                '''

genome_tags = pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/genome-tags.csv")

tfdf['final_weights'] = tfdf['tf']+tfdf['final_time_weights']
tfidf = tfdf
tfidf['idf'] = idf['idf'] ### TFIDF AND TIME WEIGHTS

tfdf = tfdf.merge(genome_tags,on='tagid')
tfdf = tfdf.sort_values(by='final_weights',ascending = False)

tfidf['final_weights'] = (tfidf['tf']*tfidf['idf'])
tfidf = tfidf.merge(genome_tags,on='tagid')
tfidf = tfidf.sort_values(by='final_weights',ascending = False)

#######################################################################

'''                   Output                        '''
if(model == 'TF'):
    pprint.pprint(list(zip(tfdf.tag,tfdf.final_weights)))
elif(model == 'TF-IDF'):
    pprint.pprint(list(zip(tfidf.tag,tfidf.final_weights)))
else:
    print("Wrong model entered!")
