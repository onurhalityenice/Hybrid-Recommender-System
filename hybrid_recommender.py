##############################################################
# Hybrid Recommender System
##############################################################

# Objective: To suggest 10 movies for the user whose ID is given,
# using the item-based and user-based recommender methods.

# Variables:
# movieId: unique movie number
# title: movie name
# genres: movie genre
# userid: unique user number
# rating: the rating given to the movie by the user
# timestamp: evaluation date

#####################################################
# Part 1:  Preparing The Data
#####################################################

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 100)

# Reading the movie.csv file
movie = pd.read_csv("Miuul/WEEK_5/datasets/movie.csv")
movie.head()
movie.shape

# Reading the rating.csv file
rating = pd.read_csv("Miuul/WEEK_5/datasets/rating.csv")
rating.head()
rating.shape
rating["userId"].nunique()

# Merging two datasets
df = movie.merge(rating, how="left", on="movieId")
df.head(20)
df.shape

# Calculating the total number of people who voted for each movie
comment_counts = pd.DataFrame(df["title"].value_counts())

# Removing movies with less than 1000 total votes from the dataset
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape

# Creating a pivot table
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()

# Functionalization
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

#####################################################
# Part 2:  Determining the Movies Watched by the User to Suggest
#####################################################

# Choosing a random user id
random_user = 103692

# Creating a new dataframe named random_user_df consisting of observation units of the selected user
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()

# Assigning the movies voted by the selected user to a list named movies_watched
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
movies_watched

#####################################################
# Part 3:  Accessing Data and Ids of Other Users Watching the Same Movies
#####################################################

# Selecting the columns of the movies watched by the selected user from user_movie_df
# and creating a new dataframe named movies_watched_df
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head(20)

# Those who watch 60 percent or more of the movies voted by the selected user can be defined as similar users.
# Creating a list named users_same_movies from the ids of these users
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(users_same_movies)

#####################################################
# Part 4:  Determining the Users to be Suggested and Most Similar Users
#####################################################

# Filtering the movies_watched_df dataframe to find the ids of users
# that are similar to the selected user in the user_same_movies list
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape

# Creating a new corr_df dataframe with correlations between users
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df[corr_df["user_id_1"] == random_user]

# Creating a new dataframe named top_users
# by filtering out users with high correlation (over 0.65) with the selected user
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape

# Merging the top_users dataframe with the rating dataset
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()

#####################################################
# Part 5:  Calculating Weighted Average Recommendation Score and Selecting Top 5 Movies
#####################################################

# Creating a new variable named weighted_rating, which is the product of each user's corr and rating
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Creating a new dataframe named recommendation_df containing the movie id
# and the average value of the weighted ratings of all users for each movie
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# Selecting movies with a weighted rating greater than 3.5 in recommendation_df
# and sorting them by weighted rating. Saving the first 5 observations as movies_to_be_recommend
recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# Bringing the names of the 5 recommended movies
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"]

#####################################################
# Part 6:  Item-Based Recommendation
#####################################################

# Making item-based suggestions based on the name of the movie
# that the user last watched and gave the highest rating
user = 103692

# Reading datasets
movie = pd.read_csv("Miuul/WEEK_5/datasets/movie.csv")
rating = pd.read_csv("Miuul/WEEK_5/datasets/rating.csv")

# Getting the id of the movie with the most recent score
# from the movies that the user to be recommended gives 5 points
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Filtering the user_movie_df dataframe created in the User based recommendation section by the selected movie id
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]

# Finding and sorting correlation of selected movie with other movies using filtered dataframe
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Functionalization
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# Recommend 5 movies
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
movies_from_item_based[1:6].index

### THE END ###