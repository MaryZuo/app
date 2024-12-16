# reference: https://campuswire.com/c/GB46E5679/feed/969
# https://liangfgithub.github.io/Proj/MovieRecommendApp.html
# https://dash.plotly.com/tutorial
# https://www.youtube.com/watch?v=H16dZMYmvqo&ab_channel=Plotly

import pandas as pd
import requests
import numpy as np

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

# The list is coming from the 100 most popular movies I got from System I
top_100_popular_movieID = [2858, 260, 1196, 1210, 480, 2028, 589, 2571, 1270, 593, 1580, 1198, 608, 2762, 110, 2396, 1197, 527, 1617, 1265, 1097, 2628, 2997, 318, 858, 356, 2716, 296, 1240, 1, 1214, 2916, 457, 3578, 1200, 541, 2987, 1259, 50, 34, 2791, 780, 3175, 1193, 919, 924, 1127, 2355, 1387, 1221, 912, 1036, 1213, 1610, 377, 1291, 2000, 1136, 3114, 1307, 1704, 1721, 1968, 648, 2599, 32, 3793, 2174, 2797, 2918, 2291, 2959, 3471, 590, 1374, 1394, 2683, 592, 1784, 1573, 1304, 3418, 223, 380, 2706, 1225, 1584, 1527, 3481, 1923, 750, 2699, 39, 21, 1393, 2804, 588, 2406, 1220, 733]

# Create a function named myIBCF
def myIBCF(new_user_ratings):
    # load the similarity matrix
    S = pd.read_csv('https://github.com/MaryZuo/CS598_Project4/raw/refs/heads/main/S_sub_100_pd_top_30.csv', index_col=0)
    #print(S.head())

    # generate newuser
    df_newuser = pd.DataFrame(columns=S.columns.to_list())
    df_newuser.loc[0] = np.nan
    for key, value in new_user_ratings.items():
        df_newuser['m' + str(key)] = value
    newuser = df_newuser.loc[0]
    #print(newuser)
        
    # initialize an empty dictionary to store predictions for unrated movies
    predictions = {}
    # iterate over all movies
    for i in S.columns:
        # skip movies that the user has already rated
        if not pd.isna(newuser[i]):
            continue
            
        # find similar movies
        similar_movies = S.loc[i].dropna()
        # find moives rated by the user, and exist in the similar_movies
        rated_by_user = newuser[similar_movies.index].dropna()

        if rated_by_user.empty:
            continue
            
        # compute the prediction for movie i:
        numerator = (similar_movies[rated_by_user.index] * rated_by_user).sum()
        denominator = similar_movies[rated_by_user.index].sum()

        if denominator == 0:
            continue

        predictions[i] = numerator / denominator

    #print(predictions)

    # sort the predictions
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    #print(sorted_predictions)
    sorted_movies = [x[0] for x in sorted_predictions]
    #print(sorted_movies)
    len_sorted_movies = len(sorted_movies)
    #print(len_sorted_movies)
    if len_sorted_movies < 10:
        #print('len of sorted movies are smaller than 10')
        df_movie_ranking_system1 = pd.read_csv('https://github.com/MaryZuo/CS598_Project4/raw/refs/heads/main/df_movie_ranking_system1.csv')
        # find the list of movies that has been rated by user
        rated_by_user_all = newuser.dropna().index.to_list()
        #print(rated_by_user_all)
        # the extra movies need to be added
        count = 10 - len_sorted_movies
        for column, _ in df_movie_ranking_system1.iloc[0].items():
            if count <= 0:
                break
            if column not in rated_by_user_all and column not in sorted_movies:
                sorted_movies.append(column)
                count -= 1
        
    return sorted_movies[0:10]

def get_displayed_movies():
    sub_movies = movies[movies["movie_id"].isin(top_100_popular_movieID)].copy()
    #print(sub_movies.head(10))
    # Sort DataFrame by the order in the list
    sub_movies.loc[:, "movie_id"] = pd.Categorical(sub_movies["movie_id"], categories=top_100_popular_movieID, ordered=True)
    sub_movies = sub_movies.sort_values("movie_id")
    #print(sub_movies.head(10))
    return sub_movies

def get_recommended_movies(new_user_ratings):
    sorted_movies = myIBCF(new_user_ratings)
    sorted_movies_id = [int(item[1:]) for item in sorted_movies]
    #print(sorted_movies)
    recommended_movie = movies[movies["movie_id"].isin(sorted_movies_id)].copy()
    recommended_movie.loc[:, "movie_id"] = pd.Categorical(recommended_movie["movie_id"], categories=sorted_movies_id, ordered=True)
    recommended_movie = recommended_movie.sort_values("movie_id")
    return recommended_movie
