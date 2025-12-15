# Movie Rating Prediction and Influential Factors Analysis Based on TMDB Data

## Project Overview

This project aims to analyze key factors that influence movie ratings and build predictive models to forecast movie ratings. By exploring relationships between movie features such as genre, cast, directors, release year, popularity, and runtime with ratings, this project provides insights for understanding movie rating patterns and predicting ratings for new movies.

## Team Members

* Name: Rui（isabella） wang & I-Chen（Iris）Li

## Project Structure

```
movie_rating_project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── movies_raw.csv
│   └── processed/
│       └── movies_cleaned.csv
├── results/
│   ├── ratings_descriptive_stats.csv
│   ├── genre_counts.csv
│   ├── feature_rating_correlation.csv
│   ├── predicted_ratings.csv
│   ├── feature_importance.csv
│   └── analysis_summary.txt
└── src/
    ├── get_data.py
    ├── clean_data.py
    ├── run_analysis.py
    └── visualize_results.py
```

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Data Collection

Data is collected from **TMDB (The Movie Database) API** using the `get_data.py` script. The script retrieves:

* Basic movie information (title, release date, runtime, language)
* Cast and crew information (directors, main actors)
* Ratings (average rating, vote count)
* Popularity and trending metrics

The raw data is saved in `data/raw/movies_raw.csv`.

### Example:

```bash
python src/get_data.py
```

## Data Cleaning

The `clean_data.py` script cleans the raw data:

* Drops missing values in important columns (release date, ratings)
* Converts release dates to datetime and extracts the year
* Converts string representations of lists (genres, directors, cast) into actual Python lists
* Saves the cleaned data to `data/processed/movies_cleaned.csv`

### Example:

```bash
python src/clean_data.py
```

## Data Analysis

The `run_analysis.py` script performs the following:

1. Descriptive statistics of movie ratings
2. Movie genre counts
3. Correlation analysis between numeric features (`popularity`, `vote_count`, `runtime`) and ratings
4. Predictive modeling using Linear Regression and Random Forest
5. Feature importance analysis from Random Forest
6. Exports results to `results/` folder

### Example:

```bash
python src/run_analysis.py
```

### Output files:

* `ratings_descriptive_stats.csv`: Summary statistics of ratings
* `genre_counts.csv`: Counts of movies per genre
* `feature_rating_correlation.csv`: Correlation of numeric features with ratings
* `predicted_ratings.csv`: Actual vs predicted ratings (Linear Regression and Random Forest)
* `feature_importance.csv`: Feature importance from Random Forest
* `analysis_summary.txt`: Summary of analysis results, RMSE, top genres, and feature importance

## Visualization (Optional)

The `visualize_results.py` script generates visualizations:

* Rating distribution (histogram & boxplot)
* Movie genre distribution (bar chart)
* Popularity vs rating scatter plot
* Feature importance plot

### Example:

```bash
python src/visualize_results.py
```

Generated figures are saved in `data/processed/`.

## How to Run the Project

Creating a virtual environment

```bash
python -m venv venv

venv\Scripts\activate
```

Installing the required libraries

```bash
pip install -r requirements.txt
```

1. Fetch raw data:

```bash
python src/get_data.py
```

2. Clean the data:

```bash
python src/clean_data.py
```

3. Run data analysis:

```bash
python src/run_analysis.py
```

4. (Optional) Generate visualizations:

```bash
python src/visualize_results.py
```

## Notes

* Ensure you have a valid TMDB API key in `get_data.py`.
* Python 3.9+ is recommended.
* CSV outputs are used for easy integration into reports and further analysis.
