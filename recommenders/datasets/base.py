"""
Base code for handling data sets.
"""
import gzip
import csv
from collections import defaultdict
from os.path import dirname
from os.path import join

import numpy as np


__all__ = ['load_movielens']




def load_movielens(ratings=True, movie_genres=True, movie_actors=True):
    module_path = join(dirname(__file__), 'data', 'movielens')
    if ratings:
        ratings_data = defaultdict(dict)
        with gzip.open(join(module_path, 'ratings.csv.gz'), 'rt', encoding='utf-8') as f:
            f.readline()
            for line in f:
                line = line.strip().split(',')
                ratings_data[int(line[0])][int(line[1])] = float(line[2])
    else:
        ratings_data = None

    if movie_genres:
        movie_genres_data = {}
        with gzip.open(join(module_path, 'movies.csv.gz'), 'rt', encoding='utf-8') as f:
            f.readline()
            lines = csv.reader(f)
            for line in lines:
                movie_genres_data[int(line[0])] = line[2].split('|')
    else:
        movie_genres_data = None

    if movie_actors:
        movie_actors_data = {}
        with gzip.open(join(module_path, 'actors.csv.gz'), 'rt', encoding='utf-8') as f:
            f.readline()
            lines = csv.reader(f)
            for line in lines:
                movie_actors_data[int(line[0])] = line[2].split('|')
    else:
        movie_actors_data = None
    return ratings_data, movie_genres_data, movie_actors_data
