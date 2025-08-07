from rag_pipeline import moviebot_rag
from data_preparation import movies_with_ratings
import pandas as pd


class MovieBot:
    def __init__(self, rag_model, movies_df):
        self.rag = rag_model
        self.movies_df = movies_df
        self.user_preferences = {
            "genres": [],
            "liked_movies": [],
            "disliked_movies": []
        }
        self.conversation_history = []

    def add_to_history(self, user_query, bot_response):
        self.conversation_history.append(
            {"user": user_query, "bot": bot_response})
        # Keep only the last 5 exchanges to maintain context without overwhelming
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

    def get_history_as_text(self):
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
        enhanced_query += "Please provide a helpful, accurate response about movies. If recommending movies, include titles, years, and genres."

        return enhanced_query

    def extract_preferences(self, query):
        """Extract user preferences from the query"""
        # This is a simplified version - in a real system, you would use NER or a dedicated classifier
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
            if any(keyword in query.lower() for keyword in keywords):
                if genre not in self.user_preferences["genres"]:
                    self.user_preferences["genres"].append(genre)

        # This is very basic - in a real system you would use NER to extract movie titles
        # For now we'll just check if they explicitly mention liking/disliking a movie
        query_lower = query.lower()
        if "i like" in query_lower or "i love" in query_lower or "i enjoyed" in query_lower:
            # Very simplistic - would need proper NER in real system
            for movie in self.movies_df['clean_title']:
                if movie.lower() in query_lower:
                    if movie not in self.user_preferences["liked_movies"]:
                        self.user_preferences["liked_movies"].append(movie)

        if "i dislike" in query_lower or "i hate" in query_lower or "i didn't like" in query_lower:
            for movie in self.movies_df['clean_title']:
                if movie.lower() in query_lower:
                    if movie not in self.user_preferences["disliked_movies"]:
                        self.user_preferences["disliked_movies"].append(movie)

    def respond(self, query):
        """Generate a response to the user query"""
        # Extract user preferences from the query
        self.extract_preferences(query)

        # Enhance the query with context
        enhanced_query = self.enhance_query_with_context(query)

        # Get response from RAG model
        response = self.rag.run(enhanced_query)

        # Add to conversation history
        self.add_to_history(query, response)

        return response

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


# Initialize the MovieBot
moviebot = MovieBot(moviebot_rag, movies_with_ratings)
