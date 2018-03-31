from db_helper import db_connect
import numpy as np
import math


class base:
    def __init__(self):
        self.conn = db_connect()
        self.idf_values = {}

    # Execute the query in kdb
    def exec_query(self, query):
        try:
            result = self.conn(query)
            return [r for r in result]
        except Exception as err:
            print "Query execution failed."
            print err

    # For given userid, get the movies watched by that user and when the user watched it.
    # A user has watched a movie either if he has tagged or rated the movie
    def user_movie(self, userid):
        movie_timestamp = {}

        user_movie_query = 'select movieid,timestamp from .moviedata.mltags where userid=`$"' + userid + '"'
        movie_list = self.exec_query(user_movie_query)
        for movie in movie_list:
            movie_timestamp[movie[0]] = np.array(movie[1]).tolist()

        user_movie_query = 'select movieid,timestamp from .moviedata.mlratings where userid=`$"' + userid + '"'
        movie_list += self.exec_query(user_movie_query)
        for movie in movie_list:
            movie_timestamp[movie[0]] = np.array(movie[1]).tolist()

        return movie_timestamp

    # Get the movies and details
    def movie_genre(self):
        movies_query = 'select movieid,genres,year,moviename from .moviedata.mlmovies'
        movies = self.exec_query(movies_query)
        movie_genre = {}
        movie_year = {}
        movie_name = {}
        genres = {}
        count = 0
        for movie in movies:
            movie_genre[movie[0]] = movie[1]
            genres_split = str(movie[1]).split("|")
            for genre in genres_split:
                if genre not in genres.keys():
                    genres[genre] = count
                    count += 1
            movie_year[movie[0]] = movie[2]
            movie_name[movie[0]] = movie[3]

        return movie_genre, movie_year, genres, movie_name

    # Get movies and its tags
    def movie_tag(self):
        movie_query = 'select movieid,moviename from .moviedata.mlmovies'
        movies_result = self.exec_query(movie_query)
        movies = {}
        for movie in movies_result:
            movies[movie[0]] = movie[1]

        tag_query = 'select tagId,tag from .moviedata.genome_tags'
        tags_result = self.exec_query(tag_query)
        tags = {}
        for tag in tags_result:
            tags[tag[0]] = tag[1]

        movie_tag_query = 'select movieid,tagid from (`movieid xgroup select movieid,tagid from .moviedata.mltags)'
        movie_tag_result = self.exec_query(movie_tag_query)
        movie_tag = {}
        for movie in movie_tag_result:
            movie_tag[movie[0]] = np.array(movie[1]).tolist()

        return movie_tag, movies, tags

    # idf - In the database, log(total no. of tables / no. of tables having that value)
    def idf(self, database, value):
        total = len(database)

        # Since for a database, idf of a value is same, it can be stored and used. No need to calculate again
        # Return idf if already calculated
        if value in self.idf_values.keys():
            return self.idf_values[value]

        # Calculate idf if not already done
        count = 0.0
        for table, values in database.iteritems():
            if value in values:
                count += 1.0
        self.idf_values[value] = math.log(total / count)

        return math.log(total / count)
