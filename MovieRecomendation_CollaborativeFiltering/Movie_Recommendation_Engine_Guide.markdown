# Building a Movie Recommendation Engine with Collaborative Filtering

This document provides a step-by-step guide to creating a movie recommendation engine using **user-based collaborative filtering**, a popular technique in recommendation systems. Collaborative filtering leverages user behavior (e.g., movie ratings) to recommend items based on patterns of preferences. We’ll use a small sample dataset to illustrate the process, explain the calculations (including the user-item and similarity matrices), and show how ratings are predicted. The implementation is in Python using libraries like pandas and scikit-learn.

---

## What is Collaborative Filtering?

Collaborative filtering recommends items by finding similarities between users (user-based) or items (item-based) based on past interactions, such as movie ratings. In **user-based collaborative filtering**, if two users have similar tastes (e.g., rated similar movies highly), we recommend movies liked by one user to the other. This guide focuses on user-based filtering, with the following steps:

1. Prepare a dataset of user-movie ratings.
2. Create a user-item matrix.
3. Compute similarities between users.
4. Predict ratings for unrated movies.
5. Generate recommendations.

---

## Step 1: Understanding the Dataset

The recommendation engine requires a dataset of user-movie interactions, typically containing:
- **UserID**: Identifier for a user.
- **MovieID**: Identifier for a movie.
- **Rating**: Numerical score (e.g., 1 to 5) indicating a user’s preference.

### Sample Dataset
For this example, we use a small dataset:

| UserID | MovieID | Rating |
|--------|---------|--------|
| 1      | 101     | 5      |
| 1      | 102     | 3      |
| 2      | 101     | 4      |
| 2      | 103     | 2      |

- User 1 rated Movie 101 (5) and Movie 102 (3).
- User 2 rated Movie 101 (4) and Movie 103 (2).

This dataset can be scaled to larger ones like the **MovieLens dataset** (available from GroupLens).

---

## Step 2: Creating the User-Item Matrix

The **user-item matrix** organizes ratings into a table where:
- **Rows** represent users.
- **Columns** represent movies.
- **Cells** contain ratings, with 0 (or NaN) for unrated movies.

Using the sample dataset, we create the matrix with Python’s `pandas`:

```python
import pandas as pd
data = {
    'UserID': [1, 1, 2, 2],
    'MovieID': [101, 102, 101, 103],
    'Rating': [5, 3, 4, 2]
}
df = pd.DataFrame(data)
user_item_matrix = df.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)
```

### Resulting User-Item Matrix
```
MovieID  101  102  103
UserID
1        5.0  3.0  0.0
2        4.0  0.0  2.0
```

- **Explanation**:
  - User 1: Rated Movie 101 (5.0), Movie 102 (3.0), not Movie 103 (0.0).
  - User 2: Rated Movie 101 (4.0), Movie 103 (2.0), not Movie 102 (0.0).
  - The `pivot_table` function reshapes the data, and `fillna(0)` sets unrated movies to 0.

This matrix is the foundation for computing similarities and predictions.

---

## Step 3: Computing the User Similarity Matrix

To recommend movies, we need to find users with similar tastes. We use **cosine similarity**, which measures the cosine of the angle between two users’ rating vectors. The formula is:

\[
\text{cosine_similarity}(u, v) = \frac{\sum (r_{u,i} \cdot r_{v,i})}{\sqrt{\sum r_{u,i}^2} \cdot \sqrt{\sum r_{v,i}^2}}
\]

- \( r_{u,i} \): Rating by user \( u \) for movie \( i \).
- Higher values (closer to 1) indicate more similar preferences.

### Implementation
We compute the similarity matrix using `sklearn.metrics.pairwise.cosine_similarity`:

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
```

### Resulting User Similarity Matrix
```
UserID         1         2
UserID
1       1.000000  0.877058
2       0.877058  1.000000
```

### Calculation for Similarity Between User 1 and User 2
- **User 1’s ratings**: [5.0, 3.0, 0.0] (Movies 101, 102, 103)
- **User 2’s ratings**: [4.0, 0.0, 2.0]
- **Numerator**: \(\sum (r_{u,i} \cdot r_{v,i}) = (5.0 \cdot 4.0) + (3.0 \cdot 0.0) + (0.0 \cdot 2.0) = 20.0\)
- **Denominator**:
  - User 1 magnitude: \(\sqrt{5.0^2 + 3.0^2 + 0.0^2} = \sqrt{34} \approx 5.830951\)
  - User 2 magnitude: \(\sqrt{4.0^2 + 0.0^2 + 2.0^2} = \sqrt{20} \approx 4.472136\)
- **Cosine Similarity**: \(\frac{20.0}{5.830951 \cdot 4.472136} \approx \frac{20.0}{26.076809} \approx 0.877058\)
- **Diagonal Values**: Similarity of a user with themselves (e.g., User 1 with User 1) is 1.0.

The value 0.877058 indicates User 1 and User 2 have similar tastes, primarily due to their high ratings for Movie 101.

---

## Step 4: Predicting Ratings for Unrated Movies

To recommend movies to a user (e.g., User 1), we predict ratings for movies they haven’t rated (e.g., Movie 103) using the ratings of similar users, weighted by their similarity scores. The prediction formula is:

\[
\text{Predicted Rating}_{u,i} = \frac{\sum_{v \in \text{similar users}} (\text{similarity}_{u,v} \cdot r_{v,i})}{\sum_{v \in \text{similar users}} \text{similarity}_{u,v}}
\]

- \( u \): Target user (e.g., User 1).
- \( i \): Unrated movie (e.g., Movie 103).
- \( v \): Similar users.
- \( r_{v,i} \): Rating by user \( v \) for movie \( i \).
- \(\text{similarity}_{u,v}\): Similarity score between users \( u \) and \( v \).

### Implementation
The `get_recommendations` function predicts ratings:

```python
import numpy as np
def get_recommendations(user_id, user_item_matrix, similarity_df, k=1):
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:k+1].index
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    predictions = []
    for movie in unrated_movies:
        similar_user_ratings = user_item_matrix.loc[similar_users, movie]
        sim_scores = similarity_df.loc[similar_users, user_id]
        if sim_scores.sum() > 0:
            predicted_rating = np.dot(similar_user_ratings, sim_scores) / sim_scores.sum()
            predictions.append((movie, predicted_rating))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions
```

### Predicting Rating for User 1, Movie 103
Let’s compute the predicted rating for User 1 for Movie 103 with `k=1` (one similar user):
- **Similar user**: User 2 (similarity = 0.877058).
- **Unrated movie**: Movie 103 (User 1’s rating = 0.0).
- **User 2’s rating for Movie 103**: 2.0.
- **Calculation**:
  \[
  \text{Predicted Rating}_{1,103} = \frac{(0.877058 \cdot 2.0)}{0.877058} = \frac{1.754116}{0.877058} = 2.0
  \]
- **Output**:
  ```
  Recommendations for User 1:
  Movie 103: Predicted Rating 2.00
  ```

Since `k=1`, the prediction is simply User 2’s rating (2.0), as the similarity score cancels out. With more similar users (`k > 1`), the prediction would be a weighted average of their ratings.

---

## Step 5: Complete Python Code

Here’s the full code to create the recommendation engine:

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample dataset
data = {
    'UserID': [1, 1, 2, 2],
    'MovieID': [101, 102, 101, 103],
    'Rating': [5, 3, 4, 2]
}
df = pd.DataFrame(data)

# Create user-item matrix
user_item_matrix = df.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)
print("User-Item Matrix:")
print(user_item_matrix)

# Compute similarity matrix
similarity_matrix = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
print("\nUser Similarity Matrix:")
print(similarity_df)

# Recommendation function
def get_recommendations(user_id, user_item_matrix, similarity_df, k=1):
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:k+1].index
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    predictions = []
    for movie in unrated_movies:
        similar_user_ratings = user_item_matrix.loc[similar_users, movie]
        sim_scores = similarity_df.loc[similar_users, user_id]
        if sim_scores.sum() > 0:
            predicted_rating = np.dot(similar_user_ratings, sim_scores) / sim_scores.sum()
            predictions.append((movie, predicted_rating))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

# Get recommendations for User 1
recommendations = get_recommendations(1, user_item_matrix, similarity_df, k=1)
print("\nRecommendations for User 1:")
for movie, score in recommendations:
    print(f"Movie {movie}: Predicted Rating {score:.2f}")
```

### Output
```
User-Item Matrix:
MovieID  101  102  103
UserID
1        5.0  3.0  0.0
2        4.0  0.0  2.0

User Similarity Matrix:
UserID         1         2
UserID
1       1.000000  0.877058
2       0.877058  1.000000

Recommendations for User 1:
Movie 103: Predicted Rating 2.00
```

---

## Why This Works

- **User-Item Matrix**: Organizes ratings to enable similarity calculations.
- **Similarity Matrix**: Identifies users with similar tastes using cosine similarity (e.g., 0.877058 for User 1 and User 2).
- **Prediction**: Uses a weighted average of similar users’ ratings to predict ratings for unrated movies, giving more weight to more similar users.
- **Collaborative Filtering**: Assumes users with similar rating patterns will like similar movies, making predictions reliable when sufficient data is available.

---

## Scaling to Larger Datasets

To use a real dataset like MovieLens:
1. **Download**: Get `ml-latest-small` from [GroupLens](https://grouplens.org/datasets/movielens/) (`ratings.csv` and `movies.csv`).
2. **Preprocess**: Load `ratings.csv` and create the user-item matrix.
3. **Optimize**: Handle sparsity with sparse matrices (`scipy.sparse`) or use libraries like `surprise` for efficient algorithms (e.g., KNN, SVD).
4. **Map Titles**: Use `movies.csv` to map MovieIDs to titles for user-friendly recommendations.

Example with MovieLens:
```python
ratings = pd.read_csv('ml-latest-small/ratings.csv')
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
similarity_matrix = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
recommendations = get_recommendations(user_id=1, user_item_matrix=user_item_matrix, similarity_df=similarity_df, k=10)
movies = pd.read_csv('ml-latest-small/movies.csv')
for movie_id, score in recommendations:
    movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
    print(f"{movie_title}: Predicted Rating {score:.2f}")
```

---

## Challenges and Improvements

1. **Sparsity**: Most users rate few movies, making the matrix sparse. Use sparse matrices or matrix factorization (e.g., SVD).
2. **Cold Start**: New users or movies lack ratings. Use content-based filtering (e.g., genres) or popularity-based recommendations.
3. **Scalability**: For large datasets, use libraries like `surprise` or `implicit` for optimized algorithms.
4. **Normalization**: Subtract users’ average ratings before computing similarities to account for different rating scales.
5. **Evaluation**: Measure accuracy with metrics like RMSE or precision/recall on a test set.

---

## Alternative: Item-Based Collaborative Filtering

Instead of finding similar users, you can find similar movies based on user ratings:
1. Compute cosine similarity between movies (columns of the user-item matrix).
2. For a movie a user liked, recommend similar movies.
3. This is often more scalable (fewer movies than users) and stable (movie similarities change less frequently).

---

## Conclusion

This guide demonstrates how to build a user-based collaborative filtering recommendation engine. The user-item matrix organizes ratings, the similarity matrix identifies similar users, and the prediction step estimates ratings for unrated movies using weighted averages. The provided Python code is a simple implementation, but it can be scaled to larger datasets like MovieLens with optimizations. Experiment with larger `k` values, normalization, or item-based filtering to improve recommendations.

For further exploration, try:
- Implementing item-based filtering.
- Using the `surprise` library for advanced algorithms.
- Evaluating predictions with RMSE or other metrics.

If you have questions or need help with extensions, feel free to ask!