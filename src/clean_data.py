import pandas as pd
import os

def main():
    """
       Load raw movie dataset, clean and preprocess key fields,
       then save the cleaned output as a new CSV file.

       Processing steps:
       1. Read raw dataset from ../data/raw/movies_raw.csv
       2. Remove rows missing release_date or vote_average
       3. Convert release_date column to datetime format
       4. Filter out entries with invalid or missing dates
       5. Extract year from release_date to a new 'year' column
       6. Convert string-encoded list fields (genres, directors, cast) to Python list objects
       7. Save processed dataset to ../data/processed/movies_cleaned.csv

       Output:
           movies_cleaned.csv — cleaned version of the raw dataset
    """
    df = pd.read_csv("../data/raw/movies_raw.csv")
    df = df.dropna(subset=["release_date", "vote_average"])
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date'])
    df['year'] = df['release_date'].dt.year
    import ast
    df['genres'] = df['genres'].apply(ast.literal_eval)
    df['directors'] = df['directors'].apply(ast.literal_eval)
    df['cast'] = df['cast'].apply(ast.literal_eval)

    os.makedirs("../data/processed", exist_ok=True)
    df.to_csv("../data/processed/movies_cleaned.csv", index=False)
    print("Cleaned movie data saved to data/processed/movies_cleaned.csv")

if __name__ == "__main__":
    main()
