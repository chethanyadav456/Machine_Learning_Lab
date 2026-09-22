"""
Case Study 1: Movie Recommendation System

Scenario:
An online streaming platform wants to enhance user experience by recommending movies
based on user preferences. The goal is to suggest movies that a user is likely to enjoy,
based on their past ratings and the behavior of other users.

Objective:
To develop a basic movie recommendation system using user ratings and movie metadata.
The system should be able to analyze user data and suggest suitable movies automatically.

Concept Used:
Collaborative filtering — recommendations are made based on similarity between users.
If two users have similar ratings on some movies, the system assumes they will like
similar movies in the future.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'User': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C'],
    'Movie': ['Inception', 'Avengers', 'Titanic', 'Inception',
              'Titanic', 'Avengers', 'Titanic', 'Joker'],
    'Rating': [5, 4, 3, 5, 2, 4, 5, 3]
}

df = pd.DataFrame(data)
rating_matrix = df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)
print("\nUser-Movie Matrix:\n", rating_matrix)

similarity = cosine_similarity(rating_matrix)
similarity_df = pd.DataFrame(
    similarity,
    index=rating_matrix.index,
    columns=rating_matrix.index
)
print("\nUser Similarity Matrix:\n", similarity_df)

user = 'A'
similar_users = similarity_df[user].sort_values(ascending=False).drop(user)
most_similar_user = similar_users.index[0]
movies_A = set(df[df['User'] == user]['Movie'])
movies_similar = set(df[df['User'] == most_similar_user]['Movie'])

recommendations = movies_similar - movies_A
print(f"\nRecommended Movies for {user}: {recommendations}")

# Evaluation:
# - The recommendation system suggests movies to User A based on the preferences of
#   the most similar user.
# - Effectiveness can be measured by user feedback or metrics such as precision, recall,
#   or RMSE if larger datasets are used.

# Conclusion:
# This case study demonstrates a simple Movie Recommendation System. By using user
# ratings and similarity analysis, we can predict and suggest movies that users are
# likely to enjoy. This approach forms the foundation of real-world recommendation
# systems used by Netflix, Amazon Prime, and other platforms.
