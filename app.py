import streamlit as st
import pandas as pd
import requests
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.frequent_patterns import apriori

# TMDB API Configuration
TMDB_API_KEY = "2a1711beb86fc66c979692d81988f512"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Helper function to clean movie titles
def clean_title(movie_title):
    import re
    return re.sub(r'\s\(\d{4}\)', '', movie_title)

# Cache data to reduce redundant computations
@st.cache_data
def load_data():
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    ratings = ratings[ratings['rating'] >= 3.5]

    # Calculate average ratings for each movie
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.rename(columns={'rating': 'avg_rating'}, inplace=True)

    # Merge ratings with movies
    movies = movies.merge(avg_ratings, on='movieId', how='left')

    # Prepare transactions for FP-Growth
    merged = movies.merge(ratings, on='movieId', how='inner')
    merged.drop(columns=['rating', 'timestamp', 'genres'], inplace=True)
    transactions = merged.groupby(by="userId")["title"].apply(list).tolist()
    
    # Create a mapping of movie titles to average ratings
    title_to_rating = dict(zip(movies['title'], movies['avg_rating']))

    return transactions, title_to_rating

@st.cache_data
def generate_rules(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)


    apriori_frequent_itemsets = apriori(df, min_support=0.01,use_colnames=True,max_len=2)
   # frequent_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True, max_len=2)
    rules = association_rules(apriori_frequent_itemsets, metric="lift", min_threshold=0.01)
    return rules

# Fetch movie poster from TMDB
@st.cache_data
def fetch_movie_image(movie_title):
    try:
        cleaned_title = clean_title(movie_title)
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": cleaned_title,
            "language": "en-US"  # Ensures results are in English
        }
        response = requests.get(url, params=params)
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path', None)
            if poster_path:
                return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
        return "https://via.placeholder.com/500x750?text=No+Image+Available"
    except Exception as e:
        st.error(f"Error fetching image for {movie_title}: {e}")
        return "https://via.placeholder.com/500x750?text=Error+Fetching+Image"

# Function to display movie cards with ratings
# Function to generate star ratings
def generate_star_rating(rating, max_stars=5):
    if pd.isna(rating):
        return "No Rating"
    full_stars = int(rating)  # Number of filled stars
    empty_stars = max_stars - full_stars  # Number of empty stars
    return f"{'★' * full_stars}{'☆' * empty_stars}"

# Function to display movie cards with star ratings
def display_cards(recommendations, title_to_rating):
    # CSS for the cards
    st.markdown("""
    <style>
    .movie-card {
        background-color: #0e0e0e;
        border-radius: 8px;
        padding: 16px;
        width: 130px;
        margin: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .movie-card img {
        width: 100%;
        height: 150px;
        object-fit: cover;
        border-radius: 8px;
    }
    .movie-card p {
        font-size: 12px;
        color: #fefefe;
        margin: 8px 0;
    }
    .movie-card .lift-score {
        color: #777;
    }
    .movie-card .stars {
        font-size: 16px;
        color: #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create a horizontal layout using st.columns
    cols = st.columns(5)  # Adjust the number of columns as needed

    for idx, (_, row) in enumerate(recommendations.iterrows()):
        consequents = list(row['consequents'])
        lift = row['lift']
        movie_title = consequents[0]
        rating = title_to_rating.get(movie_title, None)  # Get the average rating
        star_rating = generate_star_rating(rating)  # Generate star-based rating

        # Fetch images with delay to avoid overwhelming TMDB API
        image_url = fetch_movie_image(movie_title)
        time.sleep(0.5)  # Introduce a delay of 0.5 seconds

        # Use the appropriate column
        with cols[idx % 5]:
            st.markdown(f"""
            <div class="movie-card">
                <img src="{image_url}" alt="Movie Poster">
                <p><b>{movie_title}</b></p>
                <p class="lift-score">Lift Score: {lift:.2f}</p>
                <p class="stars">{star_rating}</p>
            </div>
            """, unsafe_allow_html=True)


# Streamlit UI setup
st.title("🎥 Movie Recommendation System with Apriori")
st.write("Enter a movie title below to get recommendations for similar movies.")

# Load data and generate rules
transactions, title_to_rating = load_data()
rules = generate_rules(transactions)

# Input from user
movie_title = st.text_input("Enter a movie title (e.g., 'Superman (1978)'):")

# Process user input and display recommendations
if movie_title:
    try:
        # Filter rules for the selected movie
        filtered_rules = rules[rules["antecedents"].apply(lambda x: movie_title in str(x))]
        recommendations = (filtered_rules
                           .groupby(['antecedents', 'consequents'])[['lift']]
                           .max()
                           .sort_values(by='lift', ascending=False)
                           .reset_index()
                           .head(15))
        
        if not recommendations.empty:
            st.write(f"### Recommendations for '{movie_title}':")
            display_cards(recommendations, title_to_rating)
        else:
            st.warning("No recommendations found for the entered movie.")
    except Exception as e:
        st.error(f"An error occurred: {e}") 
