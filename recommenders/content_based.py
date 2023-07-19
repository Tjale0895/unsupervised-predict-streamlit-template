"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from numpy import load

# Importing data
movies = pd.read_csv('resources/data/movies_sub.csv')

movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.
    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.
    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.
    """
    
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
# 
# 
# 

data = data_preprocessing(27000)
# Instantiating and generating the count matrix
count_vec = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=1, stop_words='english')
count_matrix = count_vec.fit_transform(data['combined_features'])
count_matrix = count_matrix.astype("float32")
cosine_sim = cosine_similarity(count_matrix, count_matrix) 
 
   

# ... (Previous code)

def content_model(movie_list, top_n=10):
    valid_movies = [movie for movie in movie_list if movie in data['title'].values]
    if not valid_movies:
        raise ValueError("One or more movies in the list are not present in the dataset.")

    # Get indices of the movies in the movie_list
    movie_indices = data[data['title'].isin(valid_movies)].index.tolist()

    # Check if the subset of movies is not empty
    if not movie_indices:
        raise ValueError("None of the movies in the list were found in the dataset.")

    # Use CountVectorizer on the 'combined_features' column
    count_vec = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
    count_matrix = count_vec.fit_transform(data['combined_features'])
    count_matrix = count_matrix.astype("float32")

    # Calculate similarity scores for the selected movies
    selected_cosine_sim = cosine_sim[movie_indices, :]

    # Combine similarity scores for selected movies and sort in descending order
    combined_scores = selected_cosine_sim.sum(axis=0)
    sorted_indices = np.argsort(combined_scores)[::-1]

    # Exclude chosen movies from top recommendations
    top_indices = [idx for idx in sorted_indices if idx not in movie_indices]

    # Check if enough top recommendations are available
    if len(top_indices) < top_n:
        raise ValueError("Not enough movies to generate top recommendations.")

    # Get the top-n recommended movies
    recommended_movies = data.iloc[top_indices[:top_n]]['title'].tolist()
    return recommended_movies
