#!usr/bin/python

from __future__ import division
import numpy as np
import operator
from base import base
import pandas as pd

base = base()


class PPR:
    def __init__(self):
        self.teleport = []
        self.movie_genre = {}
        self.seeds_weight = {}
        self.movie_name = {}
        self.movies_genres_matrix= []
        self.genres = {}

    def movie_movie_similarity(self, userid):
        # Movies watched by the user and date watched
        seeds = base.user_movie(userid)
        seeds = sorted(seeds.items(), key=operator.itemgetter(1), reverse=True)
        self.seeds_weight = {}
        count = len(seeds)
        for seed in seeds:
            self.seeds_weight[seed[0]] = count
            count -= 1

        self.movie_genre, movie_year, self.genres, self.movie_name = base.movie_genre()

        # movie_tag, movies, tags = base.movie_tag()

        self.movies_genres_matrix = np.zeros(shape=(len(movie_year), len(self.genres)), dtype=float)

        movie_index = {n: movie for movie, n in enumerate(self.movie_genre.keys())}
        # tag_index = {n: tag for tag, n in enumerate(tags)}

        # Create movie-genre matrix using count
        for movie, genre in self.movie_genre.iteritems():
            for g in self.genres:
                if g in genre:
                    self.movies_genres_matrix[movie_index[movie]][self.genres[g]] += 1

        # Create movie-movie matrix
        D = np.array(self.movies_genres_matrix)
        D_T = np.transpose(D)
        movie_movie = np.dot(D, D_T)

        # Create teleportation matrix
        self.teleport = np.zeros(shape=(len(movie_index), 1), dtype=float)
        for seed in self.seeds_weight.keys():
            self.teleport[movie_index[seed]] = self.seeds_weight[seed]

        return movie_movie

    def calculate_pagerank(self, movie_movie):
        alpha = 0.85
        err = 0.001

        # Column normalize movie-movie matrix.
        # This matrix is transition matrix
        movie_movie_norm = movie_movie / movie_movie.sum(axis=0)

        size = movie_movie_norm.shape[0]
        t = np.array(self.teleport)
        pagerank = np.ones(size)
        prev = np.zeros(size)

        # Calculate pagerank
        while np.sum(np.abs(pagerank - prev)) > err:
            prev = pagerank
            pagerank = ((1 - alpha) * np.dot(movie_movie_norm, pagerank)) + (alpha * t)

        movie_pagerank = pd.DataFrame(columns=['movies', 'movienames', 'genres', 'pagerank'])
        movie_pagerank['movies'] = self.movie_genre.keys()
        movie_pagerank['movienames'] = self.movie_name.values()
        movie_pagerank['genres'] = self.movie_genre.values()
        movie_pagerank['pagerank'] = pagerank
        movie_pagerank = movie_pagerank.sort_values(by='pagerank', ascending=False)

        # Get the top 5 movies
        top_movies = movie_pagerank[(-movie_pagerank.movies.isin(self.seeds_weight.keys()))]
        top_movies.set_index('movies', inplace=True)
        return top_movies.head(n=5).reset_index()


def main():
    print "Enter userid:"
    userid = raw_input()

    ppr = PPR()

    movie_movie = ppr.movie_movie_similarity(userid)
    top_movies = ppr.calculate_pagerank(movie_movie)
    print "Top 5 movies:"
    print top_movies


if __name__ == "__main__":
    main()