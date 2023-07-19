"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random 
import scipy as sp
import operator # <-- Convienient item retrieval during iteration
import heapq # <-- Efficient sorting of large lists

# Imported for our sanity
import warnings
warnings.filterwarnings('ignore')


# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')
ratings.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

# Merge the datasets
rate = pd.merge(ratings[['userId','movieId','rating']],movies_df[['title',"movieId"]],on = "movieId")
util_matrix = rate.pivot_table(index=['title'], columns=['userId'],values='rating')  
# Normalize each row (a given user's ratings) of the utility matrix
util_matrix_norm = util_matrix.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
# Fill Nan values with 0's, transpose matrix, and drop users with no ratings
util_matrix_norm.fillna(0, inplace=True)
util_matrix_norm = util_matrix_norm.T
util_matrix_norm = util_matrix_norm.loc[:, (util_matrix_norm != 0).any(axis=0)]
# Save the utility matrix in scipy's sparse matrix format
util_matrix_sparse = sp.sparse.csr_matrix(util_matrix_norm.values)

# Compute the similarity matrix using the cosine similarity metric
movie_similarity = cosine_similarity(util_matrix_sparse.T)
# Save the matrix as a dataframe to allow for easier indexing  
movie_sim_df = pd.DataFrame(movie_similarity,index = util_matrix_norm.columns,columns = util_matrix_norm.columns)
        




# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

   #select movie 1
    if movie_list[0] not in movie_sim_df.columns:
        movie_1 = pd.DataFrame()
    else:
        movie_1 = pd.DataFrame(movie_sim_df[movie_list[0]])
        movie_1= movie_1.reset_index()
        movie_1['similarity']= movie_1[movie_list[0]]
        movie_1=pd.DataFrame(movie_1,columns=['title','similarity'])
    #select movie 2
    if movie_list[1] not in movie_sim_df.columns:
        movie_2= pd.DataFrame()
    else:
        movie_2 = pd.DataFrame(movie_sim_df[movie_list[1]]) 
        movie_2= movie_2.reset_index()
        movie_2['similarity']= movie_2[movie_list[1]]
        movie_2=pd.DataFrame(movie_2,columns=['title','similarity'])
    #select movie 3
    if movie_list[2] not in movie_sim_df.columns:
        movie_3= pd.DataFrame()
    else:
        movie_3 = pd.DataFrame(movie_sim_df[movie_list[2]])
        movie_3= movie_3.reset_index()
        movie_3['similarity']= movie_3[movie_list[2]]
        movie_3=pd.DataFrame(movie_3,columns=['title','similarity'])

    finalmovies= pd.concat([movie_1,movie_2,movie_3])
    if finalmovies.empty:
        reco=rate.groupby('title').mean().sort_values(by='rating', ascending=False).index[:top_n].to_list()
        recommended_movies=random.sample(reco, top_n)
    else:
        recommended_movies=finalmovies.sort_values('similarity',ascending=False)
        recommended_movies = recommended_movies[~(recommended_movies['title'].isin(movie_list))]
        recommended_movies=list(recommended_movies[0:top_n]['title'])
    return recommended_movies 