import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import Counter
import os


def load_data(path: str = "../data/processed/movies_cleaned.csv") -> DataFrame:
    """
    Load the cleaned movie dataset.

    Parameters:
        path (str): File path to the cleaned movie CSV file.

    Returns:
        pandas.DataFrame: Processed movie metadata containing
        ratings, genres, directors, cast, runtime, popularity, etc.
    """
    return pd.read_csv(path)


def descriptive_statistics(
    df: DataFrame,
    save_path: str = "../results/ratings_descriptive_stats.csv"
) -> None:
    """
    Generate descriptive statistics of movie ratings and save to CSV.

    Parameters:
        df (DataFrame): Movie dataset containing vote_average column.
        save_path (str): Path for saving result CSV.

    Output:
        ratings_descriptive_stats.csv
    """
    desc_stats = df['vote_average'].describe()
    desc_stats.to_csv(save_path)
    print("✔ Descriptive statistics saved.")


def genre_distribution(
    df: DataFrame,
    save_path: str = "../results/genre_counts.csv"
) -> DataFrame:
    """
    Count frequency of movie genres and export results.

    Parameters:
        df (DataFrame): Movie dataset containing genre lists.
        save_path (str): Output file path.

    Output:
        genre_counts.csv
    """
    genre_list = sum(df['genres'].apply(eval).tolist(), [])
    genre_counts = Counter(genre_list)

    genre_df = pd.DataFrame(genre_counts.items(), columns=['genre', 'count']) \
        .sort_values(by='count', ascending=False)

    genre_df.to_csv(save_path, index=False)
    print("✔ Genre distribution saved.")
    return genre_df


def correlation_analysis(
    df: DataFrame,
    save_path: str = "../results/feature_rating_correlation.csv"
) -> DataFrame:
    """
    Compute correlation matrix between features and movie ratings.

    Features used:
        - popularity
        - vote_count
        - runtime

    Parameters:
        df (DataFrame): Movie dataset.
        save_path (str): CSV output path.

    Output:
        feature_rating_correlation.csv
    """
    numeric_features = ['popularity', 'vote_count', 'runtime']
    corr_df = df[numeric_features + ['vote_average']].corr()
    corr_df.to_csv(save_path)
    print("✔ Correlation matrix saved.")
    return corr_df


def rating_prediction_model(
    df: DataFrame,
    save_pred: str = "../results/predicted_ratings.csv",
    save_importance: str = "../results/feature_importance.csv"
) -> Tuple[float, float, DataFrame]:
    """
    Train two models (Linear Regression & Random Forest) to predict movie rating.
    Export RMSE results, predicted values, and feature importance.

    Steps:
        1. Train-test split (20% test)
        2. Train Linear Regression & Random Forest models
        3. Evaluate via RMSE
        4. Save predicted rating & feature importance

    Outputs:
        predicted_ratings.csv
        feature_importance.csv
    """
    df_model = df.dropna(subset=['popularity', 'vote_count', 'runtime'])
    X = df_model[['popularity', 'vote_count', 'runtime']]
    y = df_model['vote_average']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    pred_df = pd.DataFrame({
        'title': df_model.iloc[X_test.index]['title'].values,
        'actual_rating': y_test,
        'lr_predicted_rating': y_pred_lr,
        'rf_predicted_rating': y_pred_rf
    })
    pred_df.to_csv(save_pred, index=False)

    feat_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    feat_df.to_csv(save_importance, index=False)

    print(f"✔ Rating prediction completed. LR RMSE={lr_rmse:.2f}, RF RMSE={rf_rmse:.2f}")
    return lr_rmse, rf_rmse, feat_df


def write_summary(
    lr_rmse: float,
    rf_rmse: float,
    genre_df: DataFrame,
    feat_df: DataFrame,
    save_path: str = "../results/analysis_summary.txt"
) -> None:
    """
    Generate summary text file including:
        - RMSE of both prediction models
        - Feature importance ranking
        - Top 10 movie genres

    Output:
        analysis_summary.txt
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Summary of Movie Rating Analysis and Prediction\n\n")
        f.write(f"Linear Regression RMSE: {lr_rmse:.2f}\n")
        f.write(f"Random Forest  RMSE: {rf_rmse:.2f}\n\n")

        f.write("Feature Importance:\n")
        for feat, imp in feat_df.values:
            f.write(f"{feat}: {imp:.3f}\n")

        f.write("\nTop 10 Genres:\n")
        for genre, count in genre_df.head(10).values:
            f.write(f"{genre}: {count}\n")

    print("✔ Summary report generated.")


if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)

    df = load_data()
    descriptive_statistics(df)
    genre_df = genre_distribution(df)
    corr = correlation_analysis(df)
    lr_rmse, rf_rmse, feat_df = rating_prediction_model(df)
    write_summary(lr_rmse, rf_rmse, genre_df, feat_df)
