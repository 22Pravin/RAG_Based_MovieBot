# app.py
from flask import Flask, render_template, request, jsonify
import os
import json
import pandas as pd
from dotenv import load_dotenv
from prompt_templates import create_movie_recommendation_prompt, create_movie_information_prompt, create_general_prompt

# Import RAG components
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize RAG pipeline with custom prompts


def initialize_rag_pipeline():
    print("Initializing RAG pipeline...")

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not set in environment variables. Please add it to your .env file.")

    # Load movie documents
    if not os.path.exists('C:\D\CU\Change.AI\movie_documents.json'):
        raise FileNotFoundError(
            "Movie documents not found. Please run data_preparation.py first.")

    with open('C:\D\CU\Change.AI\movie_documents.json', 'r') as f:
        movie_docs = json.load(f)

    print(f"Loaded {len(movie_docs)} movie documents")

    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Langchain Documents
    langchain_docs = []
    for doc in movie_docs:
        langchain_docs.append(
            Document(
                page_content=doc['content'],
                metadata=doc['metadata']
            )
        )

    # Create vector store
    print("Creating vector store...")
    vector_store = LangchainFAISS.from_documents(langchain_docs, embeddings)

    # Initialize language model
    print("Initializing language model...")
    llm = OpenAI(temperature=0.7, max_tokens=512)

    # Create recommendation RAG pipeline with custom prompt
    print("Creating recommendation RAG pipeline...")
    recommendation_prompt = create_movie_recommendation_prompt()
    recommendation_rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": recommendation_prompt}
    )

    # Create information RAG pipeline with custom prompt
    print("Creating information RAG pipeline...")
    information_prompt = create_movie_information_prompt()
    information_rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": information_prompt}
    )

    # Create general RAG pipeline with custom prompt
    print("Creating general RAG pipeline...")
    general_prompt = create_general_prompt()
    general_rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": general_prompt}
    )

    print("RAG pipelines initialized successfully")

    return {
        "recommendation": recommendation_rag,
        "information": information_rag,
        "general": general_rag
    }

# Initialize MovieBot class


class MovieBot:
    def __init__(self, rag_pipelines):
        self.rag_pipelines = rag_pipelines

        # Load processed movie data
        if os.path.exists('C:\D\CU\Change.AI\processed_movies.csv'):
            self.movies_df = pd.read_csv(
                'C:\D\CU\Change.AI\processed_movies.csv')
            print(f"Loaded {len(self.movies_df)} movies from processed data")
        else:
            raise FileNotFoundError(
                "Processed movies data not found. Please run data_preparation.py first.")

        self.user_preferences = {
            "genres": [],
            "liked_movies": [],
            "disliked_movies": []
        }
        self.conversation_history = []

    def add_to_history(self, user_query, bot_response):
        """Add an exchange to the conversation history"""
        self.conversation_history.append(
            {"user": user_query, "bot": bot_response})
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

    def get_history_as_text(self):
        """Get conversation history as formatted text"""
        history_text = ""
        for exchange in self.conversation_history:
            history_text += f"User: {exchange['user']}\nMovieBot: {exchange['bot']}\n"
        return history_text

    def enhance_query_with_context(self, query):
        """Add conversation history and user preferences to the query"""
        enhanced_query = f"Based on the following conversation and preferences:\n\n"

        if self.conversation_history:
            enhanced_query += f"Conversation history:\n{self.get_history_as_text()}\n"

        if any(self.user_preferences.values()):
            enhanced_query += "User preferences:\n"
            if self.user_preferences["genres"]:
                enhanced_query += f"- Liked genres: {', '.join(self.user_preferences['genres'])}\n"
            if self.user_preferences["liked_movies"]:
                enhanced_query += f"- Liked movies: {', '.join(self.user_preferences['liked_movies'])}\n"
            if self.user_preferences["disliked_movies"]:
                enhanced_query += f"- Disliked movies: {', '.join(self.user_preferences['disliked_movies'])}\n"

        enhanced_query += f"\nUser query: {query}\n\n"

        return enhanced_query

    def extract_preferences(self, query):
        """Extract user preferences from the query"""
        query_lower = query.lower()

        # Extract genre preferences
        genre_keywords = {
            "action": ["action", "exciting", "thrilling"],
            "comedy": ["comedy", "funny", "hilarious"],
            "drama": ["drama", "dramatic", "serious"],
            "horror": ["horror", "scary", "frightening"],
            "sci-fi": ["sci-fi", "science fiction", "futuristic"],
            "romance": ["romance", "romantic", "love story"],
            "documentary": ["documentary", "real", "true story"],
            "animation": ["animation", "animated", "cartoon"]
        }

        # Check for genre preferences
        for genre, keywords in genre_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if genre not in self.user_preferences["genres"]:
                    self.user_preferences["genres"].append(genre)

        # Extract movie preferences
        like_indicators = ["like", "love", "enjoy", "favorite"]
        dislike_indicators = ["dislike", "hate", "don't like", "didn't like"]

        # Check for liked/disliked movies - basic implementation
        for _, movie_row in self.movies_df.iterrows():
            if 'clean_title' in movie_row:
                movie_title = movie_row['clean_title']
                if movie_title.lower() in query_lower:
                    # Check if it's a liked movie
                    if any(indicator in query_lower for indicator in like_indicators):
                        if movie_title not in self.user_preferences["liked_movies"]:
                            self.user_preferences["liked_movies"].append(
                                movie_title)

                    # Check if it's a disliked movie
                    elif any(indicator in query_lower for indicator in dislike_indicators):
                        if movie_title not in self.user_preferences["disliked_movies"]:
                            self.user_preferences["disliked_movies"].append(
                                movie_title)

    def determine_query_type(self, query):
        """Determine the type of query to select the appropriate RAG pipeline"""
        query_lower = query.lower()

        # Check for recommendation intent
        recommendation_keywords = ["recommend", "suggest", "what should i watch",
                                   "looking for", "similar to", "like", "good movies"]
        if any(keyword in query_lower for keyword in recommendation_keywords):
            return "recommendation"

        # Check for information intent
        information_keywords = ["tell me about", "what is", "information on", "details",
                                "plot", "who stars in", "when was", "director"]
        if any(keyword in query_lower for keyword in information_keywords):
            return "information"

        # Default to general
        return "general"

    def recommend_by_genre(self, genre, top_n=5):
        """Recommend top movies from a specific genre"""
        # Filter movies by the requested genre
        genre_movies = self.movies_df[self.movies_df['genres'].str.contains(
            genre, case=False)]

        # Sort by rating (descending) and get top_n
        if 'avg_rating' in genre_movies.columns:
            top_movies = genre_movies.sort_values(
                'avg_rating', ascending=False).head(top_n)
        else:
            top_movies = genre_movies.head(top_n)

        # Format the recommendations
        recommendations = []
        for _, movie in top_movies.iterrows():
            title = movie['clean_title']
            year = movie['year'] if 'year' in movie and not pd.isna(
                movie['year']) else 'Unknown'
            rating = f"{movie['avg_rating']:.1f}" if 'avg_rating' in movie and not pd.isna(
                movie['avg_rating']) else 'No ratings'
            genres = movie['genres'].replace('|', ', ')

            recommendation = {
                "title": title,
                "year": year,
                "rating": rating,
                "genres": genres
            }
            recommendations.append(recommendation)

        return recommendations

    def find_similar_movies(self, movie_title, top_n=5):
        """Find movies similar to the given movie title"""
        # Find the movie in our dataset
        movie = self.movies_df[self.movies_df['clean_title'].str.contains(
            movie_title, case=False)]

        if movie.empty:
            return []

        # Get the movie's genres
        movie_genres = movie.iloc[0]['genres'].split('|')

        # Find movies with similar genres
        similar_genre_movies = self.movies_df[
            self.movies_df['genres'].apply(lambda x: any(
                genre in x.split('|') for genre in movie_genres))
        ]

        # Remove the input movie
        similar_genre_movies = similar_genre_movies[
            ~similar_genre_movies['clean_title'].str.contains(
                movie_title, case=False)
        ]

        # Sort by rating and get top_n
        if 'avg_rating' in similar_genre_movies.columns:
            top_similar = similar_genre_movies.sort_values(
                'avg_rating', ascending=False).head(top_n)
        else:
            top_similar = similar_genre_movies.head(top_n)

        # Format the recommendations
        recommendations = []
        for _, similar_movie in top_similar.iterrows():
            title = similar_movie['clean_title']
            year = similar_movie['year'] if 'year' in similar_movie and not pd.isna(
                similar_movie['year']) else 'Unknown'
            rating = f"{similar_movie['avg_rating']:.1f}" if 'avg_rating' in similar_movie and not pd.isna(
                similar_movie['avg_rating']) else 'No ratings'
            genres = similar_movie['genres'].replace('|', ', ')

            recommendation = {
                "title": title,
                "year": year,
                "rating": rating,
                "genres": genres
            }
            recommendations.append(recommendation)

        return recommendations

    def respond(self, query):
        """Generate a response to the user query"""
        # Extract user preferences from the query
        self.extract_preferences(query)

        # Enhance the query with context
        enhanced_query = self.enhance_query_with_context(query)

        # Determine the type of query
        query_type = self.determine_query_type(query)

        # Get response from the appropriate RAG pipeline
        response = self.rag_pipelines[query_type].run(enhanced_query)

        # Add to conversation history
        self.add_to_history(query, response)

        return response


# Initialize RAG pipelines and MovieBot
print("Initializing MovieBot system...")
rag_pipelines = initialize_rag_pipeline()
moviebot = MovieBot(rag_pipelines)
print("MovieBot initialized and ready to chat!")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')

    if user_message.strip():
        # Get response from our MovieBot
        bot_response = moviebot.respond(user_message)
        return jsonify({'response': bot_response})

    return jsonify({'response': 'I didn\'t catch that. Could you please try again?'})


@app.route('/recommend_genre', methods=['POST'])
def recommend_genre():
    genre = request.json.get('genre', '')

    if genre.strip():
        recommendations = moviebot.recommend_by_genre(genre)
        return jsonify({'recommendations': recommendations})

    return jsonify({'error': 'Please specify a genre'})


@app.route('/similar_movies', methods=['POST'])
def similar_movies():
    movie_title = request.json.get('movie_title', '')

    if movie_title.strip():
        similar_movies = moviebot.find_similar_movies(movie_title)
        return jsonify({'similar_movies': similar_movies})

    return jsonify({'error': 'Please specify a movie title'})


@app.route('/get_genres', methods=['GET'])
def get_genres():
    # Get unique genres from the dataset
    all_genres = []
    for genres_list in moviebot.movies_df['genres'].str.split('|'):
        all_genres.extend(genres_list)

    unique_genres = sorted(list(set(all_genres)))
    return jsonify({'genres': unique_genres})


if __name__ == '__main__':
    app.run(debug=True)
