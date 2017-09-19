#!/anaconda/envs/main/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:33:09 2017

@author: shubhamgondane
"""

import pandas as pd
import numpy as np
import sys
import pprint

#genre1 = input('Enter genre 1\n')
#genre2 = input('Enter genre 2\n')
genre1 = sys.argv[1]
genre2 = sys.argv[2]
model = sys.argv[3]

movies= pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/mlmovies.csv") # Read datasets 
movie_genre1 =movies[ movies['genres'].str.contains(genre1)]
movie_genre2 =movies[ movies['genres'].str.contains(genre2)]

no_of_movies1 = movie_genre1.shape
no_of_movies2 = movie_genre2.shape

'''                     TF                      '''

def calculate_tf(movie_genre):
    mltag = pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/mltags.csv")
    movie_genre_tag = mltag.merge(movie_genre, on ='movieid')
    movie_genre_tag.drop('userid', axis=1, inplace=True)     # Drop unrelated columns
    movie_genre_tag.drop('moviename', axis=1, inplace=True)
    movie_genre_tag = movie_genre_tag.sort_values(by='timestamp',ascending=False)  # For each movie and the tags contained in it I am sorting it in order of newest tag first

    
    movie_genre_tag['time_rank_weights']=time_rank(movie_genre_tag)

    from collections import Counter
    tag_movie_pair = {k: g["tagid"].tolist() for k,g in movie_genre_tag.groupby(['movieid'])} 
    final = {}
    for key,values in tag_movie_pair.items():
        total_tags = len(values)
        c = Counter(values)
        a = {k: v / total_tags for k, v in c.items()}
        final[key] = {}
        for k,v in a.items():
            final[key][k] = v
    
# The above loop calculates the term frequency and tagid pairs for each movie seperately.

    final_df=pd.DataFrame.from_dict({(i,j): final[i][j]   #STORE ALL THE TERM FREQ CALCULATED IN A DATAFRAME
                               for i in final.keys() 
                               for j in final[i].keys()},
                           orient='index') 
    final_df.reset_index(inplace = True)
    final_df.columns = ['movie_tag','tf']
    final_df[['movie','tagid']] = final_df['movie_tag'].apply(pd.Series)
    final_df = final_df.drop('movie_tag',1)


    count_timestamps = timestamp_weight(movie_genre_tag)
    
    movie_genre_tag = movie_genre_tag.groupby(['movieid','tagid','timestamp','time_rank_weights']).size().reset_index(name='count')   #calculate the number of times a tag appears in movie and 

    tfdf = final_df.merge(count_timestamps,on='tagid')  # TF and TIME WEIGHTS
    
    tfdf = tfdf.groupby(['tagid'])['tf'].sum()
    tfdf = pd.DataFrame(tfdf)
    tfdf.reset_index(inplace = True)
    tfdf.columns = ['tagid','tf']
    tfdf = tfdf.merge(count_timestamps,on='tagid')
    
    genome_tags = pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/genome-tags.csv")
    tfdf = tfdf.merge(genome_tags,on='tagid') # combine tagid with tags
    return(tfdf)
    
'''                IDF                  '''    

def calculate_idf(movie_genre1,movie_genre2):
    mltag = pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/mltags.csv")

    total_movies_df = movie_genre1.append(movie_genre2)
    
    total_movies_df = total_movies_df.drop_duplicates(keep ='first')
    total_movies_df.drop('moviename', axis=1, inplace=True)
    total_movies_df.drop('genres', axis=1, inplace=True)
    total_movies = total_movies_df.shape
    total_movies = total_movies[0]
    movie_genre_tag = mltag.merge(total_movies_df, on ='movieid')
    movie_genre_tag = movie_genre_tag.drop_duplicates(keep = 'first')

    movie_genre_tag.drop('userid', axis=1, inplace=True)     # Drop unrelated columns
    movie_genre_tag = movie_genre_tag.sort_values(by='timestamp',ascending=False)  # For each movie and the tags contained in it I am sorting it in order of newest tag first
    movie_genre_tag['time_rank_weights']=time_rank(movie_genre_tag)
    tag_in_movie = {k: g["movieid"].tolist() for k,g in movie_genre_tag.groupby(['tagid'])} # gives me list of movies in which a particular tag appears
    count_tag_in_movie = {}
    for key,values in tag_in_movie.items():
        count_tag_in_movie[key] = len(values)
    
    count_tag_in_movies = pd.Series(count_tag_in_movie, name = 'No_of_movies')
    count_tag_in_movies.reset_index()
    count_tag_in_movies.index.names = ['tagid']
    count_tag_in_movies = pd.DataFrame(count_tag_in_movies)
    count_tag_in_movies.reset_index(inplace = True)
    
    idf = pd.merge(movie_genre_tag,count_tag_in_movies, on = 'tagid')
    
    
    count_timestamps = timestamp_weight(movie_genre_tag)

   
    
    idf['idf'] = np.log10(total_movies/idf['No_of_movies'])  # ACTUAL IDF CALCULATION
    idf = idf.groupby(['tagid'])['idf'].sum()
    idf = pd.DataFrame(idf)
    idf.reset_index(inplace = True)
    idf.columns = ['tagid','idf']
    idf = idf.merge(count_timestamps,on='tagid')
    genome_tags = pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/genome-tags.csv")

    idf = idf.merge(count_timestamps, on = 'tagid')
    idf = idf.merge(genome_tags,on='tagid')
    return(idf)
    
'''         Calculate weights for timestamps           '''  
  
def timestamp_weight(movie_genre_tag):
    tag_timestamp = {k: g["time_rank_weights"].tolist() for k,g in movie_genre_tag.groupby(['tagid'])} # to give uniform timestamp weights to each tag 
    count_timestamps= {}
    for key,values in tag_timestamp.items():
        count = len(values)
        ts = list(values)
        count_timestamps[key] = np.sum(ts)/count
        
  

    count_timestamps = pd.Series(count_timestamps,name = 'final_time_weights')
    count_timestamps.reset_index()
    count_timestamps.index.names = ['tagid']
    count_timestamps = pd.DataFrame(count_timestamps)
    count_timestamps.reset_index(inplace = True)
    
    return(count_timestamps)
    
'''          Sort and rank timestamps          '''

def time_rank(movie_genre_tag): # give sorted timestamps ranks from 1 to n
    timerank = []
    movie_tag_genre_rows = movie_genre_tag.shape
    for i in range(1,movie_tag_genre_rows[0]+1):
        timerank.append(i)

    timerank = np.divide(1,timerank)
    time_rank_weights = timerank/(np.sum(timerank,dtype = np.float64))
    return(time_rank_weights)

'''                 P-DIFF1                     '''

def p_diff1(movie_genre1,movie_genre2):
    no_of_movies1 = movie_genre1.shape
    
    R = no_of_movies1[0]
    
    total_movies_df = movie_genre1.append(movie_genre2)
    total_movies_df = total_movies_df.drop_duplicates(keep ='first')
    total_movies = total_movies_df.shape
    
    M = total_movies[0]
   
    mltag = pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/mltags.csv")
    movie_genre_tag1 = mltag.merge(movie_genre1, on ='movieid')
    movie_genre_tag1.drop('userid', axis=1, inplace=True)     # Drop unrelated columns
    movie_genre_tag1.drop('moviename', axis=1, inplace=True)
    movie_genre_tag1.drop('timestamp',axis=1,inplace =True)
    movie_genre_tag1 = movie_genre_tag1.drop_duplicates(keep = 'first')
    
    ######  r calculation #########
    r = tag_in_genre(movie_genre_tag1)
    
    r['norm'] = (r['No_of_movies']-r['No_of_movies'].min())/(r['No_of_movies'].max()-r['No_of_movies'].min()) # Normalize the r values 
    ###########  m calculation #########
    movie_genre_tag2 = mltag.merge(movie_genre2, on ='movieid')
    movie_genre_tag2.drop('userid', axis=1, inplace=True)     # Drop unrelated columns
    movie_genre_tag2.drop('moviename', axis=1, inplace=True)
    movie_genre_tag2.drop('timestamp',axis=1,inplace =True)
    
    m1 = pd.DataFrame()
    m1 = movie_genre_tag2.append(movie_genre_tag1)
    m1 = m1.drop_duplicates(keep='first')
    
    m1 = tag_in_genre(m1)
    m = r.merge(m1,on = 'tagid',how = 'left')
    m = m.fillna(0)
    ##############################
    # Final weights calculated according to the given formula with adding 0.5 as approximation
    
    m['norm'] = (m['No_of_movies_y']-m['No_of_movies_y'].min())/(m['No_of_movies_y'].max()-m['No_of_movies_y'].min())
    
    m['A'] = np.log10(((r['norm']+0.5)/(R-r['norm']+0.5))/((m['norm']-r['norm']+0.5)/(M-m['norm']-R+r['norm']+0.5)))

    m['B'] = abs((r['No_of_movies']/R)-((m['No_of_movies_x']-r['No_of_movies'])/(M-R)))
  
    m['weights'] = m['A']* m['B']
    m['No_of_movies_x'] = m['No_of_movies_x'].astype(int)
   
    genome_tags = pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/genome-tags.csv")
    r1 = m.merge(genome_tags,on='tagid')
    r1 = r1.sort_values(by='weights',ascending = False)
    pprint.pprint(list(zip(r1.tag,r1.weights))) 

'''           movie count for each tag           '''
# for each tag count number of movies it appears in 
def tag_in_genre(movie_genre_tag):
    tag_in_movie = {k: g["movieid"].tolist() for k,g in movie_genre_tag.groupby(['tagid'])} # gives me list of movies in which a particular tag appears
    count_tag_in_movie = {}
    for key,values in tag_in_movie.items():
        count_tag_in_movie[key] = len(values)
    
    count_tag_in_movies = pd.Series(count_tag_in_movie, name = 'No_of_movies')
    count_tag_in_movies.reset_index()
    count_tag_in_movies.index.names = ['tagid']
    count_tag_in_movies = pd.DataFrame(count_tag_in_movies)
    count_tag_in_movies.reset_index(inplace = True)
    return(count_tag_in_movies)

'''                  P-DIFF2                     '''

def p_diff2(movie_genre1,movie_genre2):
    no_of_movies2 = movie_genre2.shape
    
    R = no_of_movies2[0]
    total_movies_df = movie_genre1.append(movie_genre2)
    total_movies_df = total_movies_df.drop_duplicates(keep ='first')
    total_movies = total_movies_df.shape
    
    intersection = movie_genre1.merge(movie_genre2)
    intersection_count = intersection.shape
    
    M = total_movies[0]
################# r calculation #########################
    mltag = pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/mltags.csv")
    movie_genre_tag1 = mltag.merge(movie_genre1, on ='movieid')
    movie_genre_tag1.drop('userid', axis=1, inplace=True)     # Drop unrelated columns
    movie_genre_tag1.drop('moviename', axis=1, inplace=True)
    movie_genre_tag1.drop('timestamp',axis=1,inplace =True)
    r1 = tag_in_genre(movie_genre_tag1)

    df1 = movie_genre_tag1[['tagid']]

    
    
    movie_genre_tag2 = mltag.merge(movie_genre2, on ='movieid')
    movie_genre_tag2.drop('userid', axis=1, inplace=True)     # Drop unrelated columns
    movie_genre_tag2.drop('moviename', axis=1, inplace=True)
    movie_genre_tag2.drop('timestamp',axis=1,inplace =True)
    m1 = tag_in_genre(movie_genre_tag2)
    m1 = m1.merge(df1,on='tagid')
    m1 = m1.drop_duplicates(keep='first')
    
    r = m1.merge(r1,on = 'tagid',how = 'outer')
    r = r.fillna(0)    
    r['No_of_movies_x'] = (no_of_movies2[0]-r['No_of_movies_x'])
    r['norm'] = (r['No_of_movies_x']-r['No_of_movies_x'].min())/(r['No_of_movies_x'].max()-r['No_of_movies_x'].min())
##############################################################
    intersection_tag = mltag.merge(intersection,on='movieid')
    intersection_tag.drop('userid', axis=1, inplace=True)     # Drop unrelated columns
    intersection_tag.drop('moviename', axis=1, inplace=True)
    intersection_tag.drop('timestamp',axis=1,inplace =True)
    
    m1 = tag_in_genre(intersection_tag)
    m1 = m1.merge(df1,on='tagid')
    m1 = m1.drop_duplicates(keep='first')
    
    m = m1.merge(r1,on = 'tagid',how = 'outer')
    m = m.fillna(0)    
    m['No_of_movies_x'] = (intersection_count[0]-m['No_of_movies_x']) 
    m['norm'] = (m['No_of_movies_x']-m['No_of_movies_x'].min())/(m['No_of_movies_x'].max()-m['No_of_movies_x'].min())

    ###### weights calculation using the formula ###########
    Comp_A = np.log(((r['norm']+0.5)/(R-r['norm']+0.5))/((abs(m['norm']-r['norm']+0.5))/(M-m['norm']-R+r['norm']+0.5)))
    Comp_B = (r['norm']/R)-abs(((m['norm']-r['norm'])/(M-R)))
    w = pd.DataFrame()
    w['weights'] = Comp_A * Comp_B
    r1 = r1.join(w,how ='left')
    genome_tags = pd.read_csv("/Users/shubhamgondane/Desktop/MWDB/Project/phase1_dataset/genome-tags.csv")
    r1 = r1.merge(genome_tags,on='tagid')
    r1 = r1.sort_values(by='weights',ascending = False)
    pprint.pprint(list(zip(r1.tag,r1.weights)))

##############################################################
genre1_tf = calculate_tf(movie_genre1)
genre2_tf = calculate_tf(movie_genre1)
idf = calculate_idf(movie_genre1,movie_genre2) 
genre1_tf['weights1'] = (genre1_tf['tf']+genre1_tf['final_time_weights'])*idf['idf']
genre2_tf['weights2'] = (genre2_tf['tf']+genre2_tf['final_time_weights'])*idf['idf']
go=genre1_tf.dropna()
go2=genre2_tf.dropna()
go2 = go2.sort_values(by ='weights2',ascending =False)
go = go.sort_values(by = 'weights1',ascending = False)
vec1 = go[['tagid','weights1']]
vec2 = go2[['tagid','weights2']]
df = vec1.append(vec2)
df = df.fillna(0)
df['distance'] = df['weights1'] - df['weights2'] # manhattan distance
#print(df)
d = df['distance'].sum()
d = abs(d)
#print(d)

#############################################################
'''            Output              '''

if(model == 'TF-IDF-DIFF'):
    pprint.pprint(list(zip(go.tag,go.weights1)))
    #pprint.pprint(list(zip(go2.tag,go2.weights2)))
    print("Distance between genre1 and genre2",d)
elif(model == 'P-DIFF1'):
    p_diff1(movie_genre1,movie_genre2)    
elif(model == 'P-DIFF2'):
    p_diff2(movie_genre1,movie_genre2)    
else:
    print("Wrong model entered!")   
    


    
    
    
    
    
    
