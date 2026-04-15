import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score


MIN_YEAR = 2000


def load_task1_df_with_weather():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data_processed" / "task1_race_driver_podium.csv"
    print(f"Loading Task1 data from: {data_path}")
    df = pd.read_csv(data_path)
    df["year"] = df["year"].astype(int)

    # merge race-level weather
    race_weather = load_race_weather_map(project_root)
    df = df.merge(race_weather, on="raceId", how="left")

    return df, project_root


def load_race_weather_map(project_root: Path) -> pd.DataFrame:
    core_dir = project_root / "data_raw" / "f1_core"
    weather_dir = project_root / "data_raw" / "f1_weather"

    races_path = core_dir / "races.csv"
    weather_path = weather_dir / "weather_features_v4.csv"

    races = pd.read_csv(races_path)
    weather = pd.read_csv(weather_path)

    races = races[races["year"] >= MIN_YEAR].copy()

    if "datetime" in weather.columns:
        weather["datetime_parsed"] = pd.to_datetime(weather["datetime"], errors="coerce")
        weather["year"] = weather["datetime_parsed"].dt.year
    else:
        raise ValueError("weather_features_v4.csv missing 'datetime' column")

    weather["round"] = weather["round"].astype(int)
    races["round"] = races["round"].astype(int)

    merged = races.merge(
        weather[["year", "round", "name", "temperature", "precipitation", "windspeed"]],
        on=["year", "round", "name"],
        how="left",
    )

    race_weather = merged[["raceId", "temperature", "precipitation", "windspeed"]]
    return race_weather


def split_by_year(df):
    train_years = list(range(2000, 2015))
    val_years = list(range(2015, 2019))
    test_years = list(range(2019, 2024))  # 2019-2023

    train = df[df["year"].isin(train_years)].copy()
    val = df[df["year"].isin(val_years)].copy()
    test = df[df["year"].isin(test_years)].copy()

    print(f"Train samples: {len(train)} (years {min(train_years)}-{max(train_years)})")
    print(f"Val   samples: {len(val)} (years {min(val_years)}-{max(val_years)})")
    print(f"Test  samples: {len(test)} (years {min(test_years)}-{max(test_years)})")

    return train, val, test


def race_level_metrics(df_with_score: pd.DataFrame, score_col: str):
    hits = 0
    total_true_podium = 0
    top1_correct = 0
    num_races = 0

    for race_id, g in df_with_score.groupby("raceId"):
        num_races += 1
        g = g.sort_values(score_col, ascending=False)

        pred_top3 = g.head(3)
        pred_top3_drivers = set(pred_top3["driverId"].tolist())

        true_podium = g[g["is_podium"] == 1]
        true_podium_drivers = set(true_podium["driverId"].tolist())

        hits += len(pred_top3_drivers & true_podium_drivers)
        total_true_podium += len(true_podium_drivers)

        if not g.empty and (g["finish_pos"] == 1).any():
            true_winner_driver = g.loc[g["finish_pos"] == 1, "driverId"].iloc[0]
            pred_top1_driver = g.iloc[0]["driverId"]
            if pred_top1_driver == true_winner_driver:
                top1_correct += 1

    recall_at3 = hits / total_true_podium if total_true_podium > 0 else 0.0
    top1_acc = top1_correct / num_races if num_races > 0 else 0.0
    return recall_at3, top1_acc


def main():
    df, project_root = load_task1_df_with_weather()

    # clean basic columns
    df = df.dropna(subset=["raceId", "driverId"])
    df["grid_num"] = pd.to_numeric(df["grid"], errors="coerce").fillna(99)
    df["is_podium"] = df["is_podium"].astype(int)

    train, val, test = split_by_year(df)

    # feature columns: grid + pre-race stats + simple weather
    feat_cols = [
        "grid_num",
        "year",
        "round",
        "prerace_points",
        "prerace_rank",
        "prev1_finish",
        "prev3_avg_finish",
        "prev3_points",
        "temperature",
        "precipitation",
        "windspeed",
    ]

    # ensure all feature columns exist; if not, fill with 0
    for col in feat_cols:
        if col not in train.columns:
            train[col] = 0.0
            val[col] = 0.0
            test[col] = 0.0

    # fill NaN with median from train
    medians = train[feat_cols].median()
    train[feat_cols] = train[feat_cols].fillna(medians)
    val[feat_cols] = val[feat_cols].fillna(medians)
    test[feat_cols] = test[feat_cols].fillna(medians)

    X_train = train[feat_cols].values
    y_train = train["is_podium"].values

    X_val = val[feat_cols].values
    y_val = val["is_podium"].values

    X_test = test[feat_cols].values
    y_test = test["is_podium"].values

    # standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Logistic Regression baseline
    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)

    # global metrics
    y_test_proba = clf.predict_proba(X_test_s)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_test_proba)
    f1 = f1_score(y_test, y_test_pred)

    print("\n=== Logistic Regression baseline (with weather) ===")
    print(f"ROC-AUC : {roc:.4f}")
    print(f"F1-score: {f1:.4f}")

    # race-level metrics (Recall@3, Top-1 champion accuracy)
    test_eval = test.copy()
    test_eval["score_logreg"] = y_test_proba
    r3, top1 = race_level_metrics(test_eval, "score_logreg")
    print(f"Recall@3 (podium hit rate): {r3:.4f}")
    print(f"Top-1 champion accuracy   : {top1:.4f}")


if __name__ == "__main__":
    main()
