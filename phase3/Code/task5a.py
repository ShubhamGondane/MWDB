from scipy import spatial
from base import base
import numpy as np
import operator

base = base()

movie_genre, movie_year, genres, movie_name = base.movie_genre()
movie_index = {n: movie for movie, n in enumerate(movie_genre.keys())}
genre_index = {n: genre for genre, n in enumerate(genres.keys())}

# Create movie-genre for tf-idf values
movie_genre_matrix = np.zeros(shape=(len(movie_genre.keys()), len(genres.keys())), dtype=float)
for movie, genres in movie_genre.items():
    genre = genres.split("|")
    for g in genre:
        tf = 1 / float(len(genre))
        idf = base.idf(movie_genre, g)
        movie_genre_matrix[movie_index[movie]][genre_index[g]] = tf * idf

input_labels = []
input_movie_label = {}
print "Enter number of labels:"
no_labels = input()
for i in range(no_labels):
    print "Enter label:"
    label = raw_input()
    input_labels.append(label)
    print "Enter movie ids for the label:"
    movies = raw_input().split(",")
    for m in movies:
        input_movie_label[m] = label

# input_movie_label = {'5259': 'Comedy', '4026': 'War', '10120': 'Drama', '5981': 'Action'}
input_movies = input_movie_label.keys()
print "Enter k:"
k = input()
count = 0
movie_label = {}
# Calculate the distance between each movie and input movies.
# For each movie, assign the label which is max of input movies based on distance between it.
for movie in movie_genre.keys():
    distance = {}
    for m in input_movies:
        distance[m] = spatial.distance.cosine(movie_genre_matrix[movie_index[movie]], movie_genre_matrix[movie_index[m]])
    min_dist = sorted(distance.items(), key=operator.itemgetter(1))
    min_dist = min_dist[0:k]
    label_count = {}
    for i in min_dist:
        label = input_movie_label[i[0]]
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1
    max_label = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
    movie_label[movie] = max_label[0][0]

label = {}
for key, value in movie_label.items():
    label.setdefault(value, []).append(key)

for key, value in label.items():
    print key, ":", "<", len(value), "movies>"
    for v in value:
        print movie_name[v], ";",
    print

print
for key, value in label.items():
    print key, ": <", len(value), "movies>"
