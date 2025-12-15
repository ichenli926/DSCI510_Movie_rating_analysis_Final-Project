import requests
import pandas as pd
import os

API_KEY = "d424193212a5a38c213154449d3c7f49"
BASE_URL = "https://api.themoviedb.org/3"

def get_movies(page=1):
    """
        Fetch a batch of popular movies from TMDB API.

        Parameters:
            page (int): Page number of the popular movie list to retrieve.
                        Each page returns up to 20 movies.

        Returns:
            dict: JSON response containing movie metadata, including title,
                  release date, popularity, and vote statistics.
    """
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
    response = requests.get(url)
    return response.json()

def get_movie_details(movie_id):
    """
        Retrieve detailed information for a specific movie including credits.

        Parameters:
            movie_id (int): Unique TMDB movie ID.

        Returns:
            dict: JSON containing extended movie info — runtime, genres,
                  cast (full list), directors (from crew data), etc.
    """
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US&append_to_response=credits"
    response = requests.get(url)
    return response.json()

def main():
    """
       Scrape movie data from TMDB API and save as raw dataset.

       Process steps:
       1. Request top 5 pages of popular movies (≈100 samples total)
       2. For each movie:
          - Retrieve detailed metadata with credits (directors + cast)
          - Extract structured fields: genres, runtime, director name(s), top 3 actors
       3. Store all extracted records as a DataFrame
       4. Save as ../data/raw/movies_raw.csv for later cleaning & analysis

       Output:
           movies_raw.csv — the unprocessed raw movie dataset
    """
    all_movies = []
    for page in range(1, 6):  # 前5页热门电影，每页20条
        data = get_movies(page)
        print(data)
        for movie in data['results']:
            details = get_movie_details(movie['id'])
            genres = [g['name'] for g in details.get('genres', [])]
            directors = [c['name'] for c in details.get('credits', {}).get('crew', []) if c['job']=='Director']
            cast = [c['name'] for c in details.get('credits', {}).get('cast', [])[:3]]  # 主要3位演员
            all_movies.append({
                "id": movie['id'],
                "title": movie['title'],
                "release_date": movie['release_date'],
                "vote_average": movie['vote_average'],
                "vote_count": movie['vote_count'],
                "popularity": movie['popularity'],
                "runtime": details.get('runtime'),
                "genres": genres,
                "directors": directors,
                "cast": cast
            })

    df = pd.DataFrame(all_movies)
    os.makedirs("../data/raw", exist_ok=True)
    df.to_csv("../data/raw/movies_raw.csv", index=False)
    print("Raw movie data saved to data/raw/movies_raw.csv")

if __name__ == "__main__":
    main()
