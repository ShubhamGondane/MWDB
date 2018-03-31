from base import base
from decisionTree import decisionTree
import numpy as np

base = base()

movie_genre, movie_year, genres, movie_names = base.movie_genre()
comedy = 1
drama = 2
fantasy =3
war = 4

movies_genres_matrix = np.zeros(shape=(len(movie_year), len(genres)), dtype=float)

movie_index = {n: movie for movie, n in enumerate(movie_genre.keys())}
# tag_index = {n: tag for tag, n in enumerate(tags)}
genre_list = []
# Create movie-genre matrix using count
for movie, genre in movie_genre.items():
    for g in genres:
        if g in genre:
            movies_genres_matrix[movie_index[movie]][genres[g]] += 1
    genre_list.append(movie_genre[movie])

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


def train(data, labels):
    model = decisionTree()
    model.train(data, labels)
    return model


def classify(x, model):
    result = model.classify(x)
    return input_labels[result]


def test(x, model):
    label = model.classify(x)
    return label


def decisionTreeClassification():
    data = []
    labels = []
    for movie in input_movies:
        data.append(movies_genres_matrix[movie_index[movie]])
        labels.append(input_labels.index(input_movie_label[movie]))

    model = train(data, labels)

    label_movie = {}
    for movie in movie_genre.keys():
        result = test(movies_genres_matrix[movie_index[movie]], model)
        label = input_labels[result]
        if label not in label_movie.keys():
            label_movie[label] = []
        label_movie[label].append(movie)

    for key, value in label_movie.items():
        print key, ": <", len(value), "movies>"
        for v in value:
            print movie_names[v], ";",
        print
    print
    for key, value in label_movie.items():
        print key, ": <", len(value), "movies>"


decisionTreeClassification()
