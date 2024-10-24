#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('spotify dataset.csv')

# Display the column names in your DataFrame
print("Column Names:", df.columns)
print(df.head())
print(df.tail())

# Find the total number of rows
total_rows = df.shape[0]

# Print the result
print("Total Rows:", total_rows)

# Feature Selection
selected_features = ['track_id', 'track_popularity', 'track_album_release_date',
                     'playlist_id', 'playlist_genre', 'playlist_subgenre',
                     'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                     'duration_ms']
df = df[selected_features]

# Data Transformation
label_encoder = LabelEncoder()
df['playlist_genre'] = label_encoder.fit_transform(df['playlist_genre'])  # Encode categorical variable 'playlist_genre'

scaler = StandardScaler()
numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                     'duration_ms']

df[numeric_features] = scaler.fit_transform(df[numeric_features])  # Standardize numerical features

# Data Splitting
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)


# In[6]:


conda install pandas


# In[8]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split  

# Assuming your data is stored in a CSV file named 'data.csv'
df = pd.read_csv('spotify dataset.csv')

# Feature Selection
selected_features = ['track_id', 'track_name', 'track_artist', 'track_popularity',
                     'track_album_id', 'track_album_name', 'track_album_release_date',
                     'playlist_name', 'playlist_id', 'playlist_genre', 'playlist_subgenre',
                     'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                     'duration_ms']
df = df[selected_features]

# Data Transformation
scaler = StandardScaler()
numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                     'duration_ms']

df[numeric_features] = scaler.fit_transform(df[numeric_features])  # Standardize numerical features

# Data Splitting
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Save the Pre-processed Data
df.to_csv('preprocessed_data.csv', index=False)
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pairplot for numeric features
sns.pairplot(df[['danceability', 'energy', 'loudness', 'valence', 'tempo']])
plt.show()


# In[10]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('spotify dataset.csv')

# Check if the 'track_name' column is present
if 'track_name' in df.columns:
   # Fill any missing values in 'track_name'
   df['track_name'].fillna('', inplace=True)

   # Create a CountVectorizer to convert text features into vectors
   vectorizer = CountVectorizer(stop_words='english')
   track_matrix = vectorizer.fit_transform(df['track_name'])

   # Calculate the cosine similarity between tracks
   cosine_sim = cosine_similarity(track_matrix, track_matrix)

   # Function to get recommendations based on cosine similarity
   def get_recommendations(track_name, cosine_sim_matrix, df):
       idx = df[df['track_name'] == track_name].index[0]
       sim_scores = list(enumerate(cosine_sim_matrix[idx]))
       sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
       sim_scores = sim_scores[1:6]  # Top 5 similar tracks (excluding itself)
       track_indices = [i[0] for i in sim_scores]
       return df['track_name'].iloc[track_indices]

# Example: Choose a track name that exists in your dataset
track_to_recommend =  'Typhoon - Original Mix'

# Check if the chosen track name exists in the DataFrame
if track_to_recommend in df['track_name'].values:
   recommendations = get_recommendations(track_to_recommend, cosine_sim, df)

   # Display the recommendations
   print(f"Top 5 Recommendations for '{track_to_recommend}':")
   print(recommendations)
else:
   print(f"The track '{track_to_recommend}' does not exist in the DataFrame.")


# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your data is stored in a CSV file named 'data.csv'
df = pd.read_csv('spotify dataset.csv')

# Feature Selection
selected_features = ['track_id', 'track_name', 'track_artist', 'track_popularity',
                     'track_album_id', 'track_album_name', 'track_album_release_date',
                     'playlist_name', 'playlist_id', 'playlist_genre', 'playlist_subgenre',
                     'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                     'duration_ms']
df = df[selected_features]

# Data Transformation
scaler = StandardScaler()
numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                     'duration_ms']

df[numeric_features] = scaler.fit_transform(df[numeric_features])  # Standardize numerical features

# Select only the numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Correlation matrix heatmap
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[ ]:




