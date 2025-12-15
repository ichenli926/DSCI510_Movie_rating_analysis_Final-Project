import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_rating_distribution(
    df: DataFrame,
    save_path: str = "../data/processed/rating_distribution.png"
) -> None:
    """
    Plot and save the distribution of movie ratings using histogram and boxplot.

    Parameters:
        df (DataFrame): Processed movie dataset containing 'vote_average' column.
        save_path (str): Output path for the generated image.

    Output:
        rating_distribution.png
    """
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    sns.histplot(df['vote_average'], bins=10, kde=True)
    plt.title("Rating Distribution Histogram")

    plt.subplot(1,2,2)
    sns.boxplot(y=df['vote_average'])
    plt.title("Rating Box Plot")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print("✔ Rating distribution plot saved.")


def plot_genre_popularity(
    df: DataFrame,
    save_path: str = "../data/processed/genre_popularity.png"
) -> DataFrame:
    """
    Count genre frequency and generate bar plot visualization.

    Parameters:
        df (DataFrame): Movie dataset where 'genres' is a list-like field.
        save_path (str): Output image save path.

    Output:
        genre_popularity.png
    """
    genre_list = sum(df['genres'].apply(eval).tolist(), [])
    genre_counts = Counter(genre_list)

    genre_df = pd.DataFrame(genre_counts.items(), columns=['genre','count']) \
                .sort_values(by='count', ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x='genre', y='count', data=genre_df)
    plt.xticks(rotation=45)
    plt.title("Distribution of Movie Genre Counts")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print("✔ Genre popularity plot saved.")
    return genre_df


def plot_popularity_vs_rating(
    df: DataFrame,
    save_path: str = "../data/processed/popularity_vs_rating.png"
) -> None:
    """
    Create a scatterplot showing relationship between popularity and rating.

    Parameters:
        df (DataFrame): Movie dataset containing 'popularity' & 'vote_average'.
        save_path (str): Output figure save location.

    Output:
        popularity_vs_rating.png
    """
    plt.figure(figsize=(10,5))
    sns.scatterplot(x='popularity', y='vote_average', data=df)
    plt.title("Movie Popularity vs. Rating")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print("✔ Popularity vs rating scatterplot saved.")


def model_rating_prediction(
    df: DataFrame,
    img_save: str = "../data/processed/feature_importance.png"
) -> Tuple[Dict[str, float], DataFrame]:
    """
    Train Linear Regression and Random Forest to predict user rating.

    Steps:
        1. Split data into train/test sets
        2. Train model and evaluate using RMSE
        3. Output feature importance and result prints

    Parameters:
        df (DataFrame): Clean movie dataset with numeric features.
        img_save (str): Output path for the feature importance plot.

    Returns:
        dict: RMSE results of LR & RF models
        DataFrame: feature importance ranking
    """
    df_model = df.dropna(subset=['popularity', 'vote_count', 'runtime'])
    X = df_model[['popularity', 'vote_count', 'runtime']]
    y = df_model['vote_average']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))
    print("Linear Regression RMSE:", lr_rmse)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
    print("Random Forest RMSE:", rf_rmse)

    # Feature importance visualization
    feat_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(8,4))
    sns.barplot(x='feature', y='importance', data=feat_df)
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(img_save)
    plt.show()
    print("✔ Feature importance plot saved.")

    return {"LR_RMSE": lr_rmse, "RF_RMSE": rf_rmse}, feat_df



if __name__ == "__main__":
    df = pd.read_csv("../data/processed/movies_cleaned.csv")

    plot_rating_distribution(df)
    genre_df = plot_genre_popularity(df)
    plot_popularity_vs_rating(df)
    model_rating_prediction(df)
