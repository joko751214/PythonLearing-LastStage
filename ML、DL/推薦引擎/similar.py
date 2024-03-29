import os
import sys
import json
import numpy as np


def calc_ps(ratings, user1, user2):
    movies = set()
    for movie in ratings[user1]:
        if movie in ratings[user2]:
            movies.add(movie)
    n = len(movies)
    if n == 0:
        return 0
    x = np.array([ratings[user1][move] for move in movies])
    y = np.array([ratings[user2][move] for move in movies])
    sx = x.sum()
    sy = y.sum()
    xx = (x**2).sum()
    yy = (y**2).sum()
    xy = (x * y).sum()
    sxx = xx - sx**2 / n
    syy = yy - sy**2 / n
    sxy = xy - sx * sy / n
    if sxx * syy == 0:
        return 0
    pearson_score = sxy / np.sqrt(sxx * syy)
    return pearson_score


def read_data(filename):
    with open(filename, 'r') as f:
        ratings = json.loads(f.read())
    return ratings


def eval_ps(ratings):
    users, psmat = list(ratings.keys()), []
    for user1 in users:
        psrow = []
        for user2 in users:
            psrow.append(calc_ps(ratings, user1, user2))
        psmat.append(psrow)
    users = np.array(users)
    psmat = np.array(psmat)
    return users, psmat


def find_similars(users, psmat, user, n_similars=None):
    user_index = np.arange(len(users))[users == user][0]
    sorted_indices = psmat[user_index].argsort()[::-1]
    similar_indices = sorted_indices[
        sorted_indices != user_index][:n_similars]
    similar_users = users[similar_indices]
    similar_scores = psmat[user_index][similar_indices]
    return similar_users, similar_scores


def main(argc, argv, envp):
    ratings = read_data('ratings.json')
    users, psmat = eval_ps(ratings)
    print(users)
    print(psmat)
    similar_users, similar_scores = find_similars(
        users, psmat, 'Alex Roberts', 3)
    print(similar_users)
    print(similar_scores)
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
