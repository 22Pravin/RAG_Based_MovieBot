import pandas as pd
import os
import requests
import zipfile
from dotenv import load_dotenv
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Download dataset (MovieLens small dataset)
dataset_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
zip_path = "ml-latest-small.zip"
extract_dir = "."  # Extract to current directory

# Create function to download the dataset


def download_dataset(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure we notice bad responses
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download completed.")
    else:
        print(f"File {save_path} already exists, skipping download.")

# Create function to extract the dataset


def extract_dataset(zip_path, extract_to):
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction completed.")
    else:
        print(f"Error: {zip_path} does not exist.")


# Download and extract the dataset
download_dataset(dataset_url, zip_path)
extract_dataset(zip_path, extract_dir)

# Load movie data
movies_df = pd.read_csv('ml-latest-small/movies.csv')
ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
links_df = pd.read_csv('ml-latest-small/links.csv')
tags_df = pd.read_csv('ml-latest-small/tags.csv')

# Merge datasets to create a comprehensive movie database
# Calculate average rating for each movie
avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
avg_ratings.rename(columns={'rating': 'avg_rating'}, inplace=True)

# Merge movies with their average ratings
movies_with_ratings = pd.merge(
    movies_df, avg_ratings, on='movieId', how='left')

# Extract year from title and create a clean title column
movies_with_ratings['year'] = movies_with_ratings['title'].str.extract(
    r'\((\d{4})\)$')
movies_with_ratings['clean_title'] = movies_with_ratings['title'].str.replace(
    r'\s*\(\d{4}\)$', '', regex=True)

# Get genre as a list
movies_with_ratings['genres_list'] = movies_with_ratings['genres'].str.split(
    '|')

# Save the processed dataset
movies_with_ratings.to_csv('processed_movies.csv', index=False)

# Create document chunks from our movie data
movie_docs = []

for _, row in movies_with_ratings.iterrows():
    movie_id = row['movieId']
    title = row['clean_title']
    year = row['year'] if not pd.isna(row['year']) else 'Unknown'
    genres = row['genres'].replace('|', ', ')
    rating = f"{row['avg_rating']:.1f}" if not pd.isna(
        row['avg_rating']) else 'No ratings yet'

    # Create a detailed document for each movie
    movie_text = f"Movie: {title} ({year})\n"
    movie_text += f"Genres: {genres}\n"
    movie_text += f"Average Rating: {rating} out of 5\n"

    # Add any tags available for this movie
    movie_tags = tags_df[tags_df['movieId'] == movie_id]
    if not movie_tags.empty:
        tags_list = movie_tags['tag'].tolist()
        movie_text += f"Tags: {', '.join(tags_list)}\n"

    # Add movie ID for retrieval purposes
    movie_text += f"MovieID: {movie_id}\n"

    movie_docs.append({
        "id": f"movie_{movie_id}",
        "content": movie_text,
        "metadata": {
            "title": title,
            "year": year,
            "genres": genres,
            "rating": rating,
            "movie_id": movie_id
        }
    })

# Save documents to a JSON file for easy loading
with open('movie_documents.json', 'w') as f:
    json.dump(movie_docs, f)
