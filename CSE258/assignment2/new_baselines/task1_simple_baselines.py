import numpy as np
import pandas as pd
from pathlib import Path

MIN_YEAR = 2000


def load_task1_df():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data_processed" / "task1_race_driver_podium.csv"
    print(f"Loading Task1 data from: {data_path}")
    df = pd.read_csv(data_path)
    df["year"] = df["year"].astype(int)
    return df, project_root


# ------------ Weather + races join ------------

def load_race_weather_map(project_root: Path) -> pd.DataFrame:
    """
    Build a mapping raceId -> (temperature, precipitation, windspeed)
    by joining races.csv with weather_features_v4.csv on (year, round, name).
    """
    core_dir = project_root / "data_raw" / "f1_core"
    weather_dir = project_root / "data_raw" / "f1_weather"

    races_path = core_dir / "races.csv"
    weather_path = weather_dir / "weather_features_v4.csv"

    races = pd.read_csv(races_path)
    weather = pd.read_csv(weather_path)

    # Only keep modern era (safety, though weather has 1950+)
    races = races[races["year"] >= MIN_YEAR].copy()

    # Parse weather year from datetime
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
    """
    df_with_score must contain:
      ['raceId', 'driverId', 'is_podium', 'finish_pos', score_col]
    Returns:
      recall_at_3, top1_accuracy
    """
    hits = 0
    total_true_podium = 0
    top1_correct = 0
    num_races = 0

    for race_id, g in df_with_score.groupby("raceId"):
        num_races += 1
        g = g.sort_values(score_col, ascending=False)

        # predicted top-3
        pred_top3 = g.head(3)
        pred_top3_drivers = set(pred_top3["driverId"].tolist())

        # true podium (finish_pos in [1,2,3])
        true_podium = g[g["is_podium"] == 1]
        true_podium_drivers = set(true_podium["driverId"].tolist())

        hits += len(pred_top3_drivers & true_podium_drivers)
        total_true_podium += len(true_podium_drivers)

        # Top-1 champion accuracy
        if not g.empty and (g["finish_pos"] == 1).any():
            true_winner_driver = g.loc[g["finish_pos"] == 1, "driverId"].iloc[0]
            pred_top1_driver = g.iloc[0]["driverId"]
            if pred_top1_driver == true_winner_driver:
                top1_correct += 1

    recall_at3 = hits / total_true_podium if total_true_podium > 0 else 0.0
    top1_acc = top1_correct / num_races if num_races > 0 else 0.0
    return recall_at3, top1_acc


# ------------ Era definition & weather bucket ------------

def get_era(year: int) -> str:
    """
    Very rough regulation eras:
      - 'pre_hybrid'   : 2000-2013
      - 'v6_hybrid'    : 2014-2021
      - 'ground_effect': 2022+
    """
    if year <= 2013:
        return "pre_hybrid"
    elif year <= 2021:
        return "v6_hybrid"
    else:
        return "ground_effect"


def bucket_precip(x: float) -> str:
    """
    Map precipitation to coarse weather buckets.
    """
    if pd.isna(x):
        return "unknown"
    if x <= 0.0:
        return "dry"
    if x < 1.0:
        return "light_rain"
    return "heavy_rain"


# ------------ Existing baselines ------------

def baseline_grid(test: pd.DataFrame):
    df = test.copy()
    df["grid_num"] = pd.to_numeric(df["grid"], errors="coerce").fillna(99)
    df["score_grid"] = -df["grid_num"]  # smaller grid -> larger score

    r3, top1 = race_level_metrics(df, "score_grid")
    print("\n=== Baseline: Grid-only ===")
    print(f"Recall@3 (podium hit rate): {r3:.4f}")
    print(f"Top-1 champion accuracy   : {top1:.4f}")


def baseline_global_podium(train: pd.DataFrame, test: pd.DataFrame):
    p_global = train["is_podium"].mean()
    df = test.copy()
    df["score_global"] = p_global  # constant score

    r3, top1 = race_level_metrics(df, "score_global")
    print("\n=== Baseline: Global podium rate (constant) ===")
    print(f"Global podium probability : {p_global:.4f}")
    print(f"Recall@3 (podium hit rate): {r3:.4f}")
    print(f"Top-1 champion accuracy   : {top1:.4f}")


def baseline_driver_podium(train: pd.DataFrame, test: pd.DataFrame):
    grouped = train.groupby("driverId")["is_podium"]
    cnt = grouped.count()
    pod = grouped.sum()
    p_driver = (pod / cnt).to_dict()
    p_global = train["is_podium"].mean()

    df = test.copy()
    df["score_driver"] = df["driverId"].map(p_driver).fillna(p_global)

    r3, top1 = race_level_metrics(df, "score_driver")
    print("\n=== Baseline: Driver podium rate ===")
    print(f"Global podium probability : {p_global:.4f}")
    print(f"Num drivers with history  : {len(p_driver)}")
    print(f"Recall@3 (podium hit rate): {r3:.4f}")
    print(f"Top-1 champion accuracy   : {top1:.4f}")


def baseline_driver_circuit_podium(train: pd.DataFrame, test: pd.DataFrame, min_count: int = 3):
    # driver-circuit level statistics
    grouped = train.groupby(["driverId", "circuitId"])["is_podium"]
    cnt = grouped.count()
    pod = grouped.sum()

    stats = pd.DataFrame({"count": cnt, "podium": pod})
    stats["rate"] = stats["podium"] / stats["count"]
    stats = stats.reset_index()

    dc_rate = stats.set_index(["driverId", "circuitId"])["rate"].to_dict()
    dc_count = stats.set_index(["driverId", "circuitId"])["count"].to_dict()

    # fallback driver-level
    g_driver = train.groupby("driverId")["is_podium"]
    d_cnt = g_driver.count()
    d_pod = g_driver.sum()
    p_driver = (d_pod / d_cnt).to_dict()
    p_global = train["is_podium"].mean()

    def score_row(row):
        key = (row["driverId"], row["circuitId"])
        if key in dc_rate and dc_count.get(key, 0) >= min_count:
            return dc_rate[key]
        d = row["driverId"]
        if d in p_driver:
            return p_driver[d]
        return p_global

    df = test.copy()
    df["score_driver_circuit"] = df.apply(score_row, axis=1)

    r3, top1 = race_level_metrics(df, "score_driver_circuit")
    print("\n=== Baseline: Driver–Circuit podium rate (min_count = {}) ===".format(min_count))
    print(f"Global podium probability : {p_global:.4f}")
    print(f"Num (driver,circuit) pairs: {len(dc_rate)}")
    print(f"Recall@3 (podium hit rate): {r3:.4f}")
    print(f"Top-1 champion accuracy   : {top1:.4f}")


def baseline_star_driver(train: pd.DataFrame, test: pd.DataFrame):
    # star drivers: rank by total number of podiums in training
    grouped = train.groupby("driverId")["is_podium"]
    pod = grouped.sum()
    pod = pod[pod > 0]

    if pod.empty:
        print("\n=== Baseline: Star-driver (no podiums found in training) ===")
        return

    max_pod = pod.max()
    # map to [0, 1]
    score_map = ((pod + 1) / (max_pod + 2)).to_dict()
    default_score = 0.0

    df = test.copy()
    df["score_star_driver"] = df["driverId"].map(score_map).fillna(default_score)

    r3, top1 = race_level_metrics(df, "score_star_driver")
    print("\n=== Baseline: Star-driver (career podium counts) ===")
    print(f"Num star drivers (podium>0): {len(score_map)}")
    print(f"Recall@3 (podium hit rate)  : {r3:.4f}")
    print(f"Top-1 champion accuracy     : {top1:.4f}")


def baseline_constructor_podium(train: pd.DataFrame, test: pd.DataFrame):
    if "constructorId" not in train.columns:
        print("\n[Warning] constructorId not found. Skipping constructor baseline.")
        return

    grouped = train.groupby("constructorId")["is_podium"]
    cnt = grouped.count()
    pod = grouped.sum()
    p_constructor = (pod / cnt).to_dict()
    p_global = train["is_podium"].mean()

    df = test.copy()
    df["score_constructor"] = df["constructorId"].map(p_constructor).fillna(p_global)

    r3, top1 = race_level_metrics(df, "score_constructor")
    print("\n=== Baseline: Constructor podium rate ===")
    print(f"Global podium probability : {p_global:.4f}")
    print(f"Num constructors          : {len(p_constructor)}")
    print(f"Recall@3 (podium hit rate): {r3:.4f}")
    print(f"Top-1 champion accuracy   : {top1:.4f}")


def baseline_circuit_podium(train: pd.DataFrame, test: pd.DataFrame):
    """
    Pure circuit baseline: learn p(podium | circuitId) from training.
    """
    grouped = train.groupby("circuitId")["is_podium"]
    cnt = grouped.count()
    pod = grouped.sum()
    p_circuit = (pod / cnt).to_dict()
    p_global = train["is_podium"].mean()

    df = test.copy()
    df["score_circuit"] = df["circuitId"].map(p_circuit).fillna(p_global)

    r3, top1 = race_level_metrics(df, "score_circuit")
    print("\n=== Baseline: Circuit-only podium rate ===")
    print(f"Global podium probability : {p_global:.4f}")
    print(f"Num circuits              : {len(p_circuit)}")
    print(f"Recall@3 (podium hit rate): {r3:.4f}")
    print(f"Top-1 champion accuracy   : {top1:.4f}")


def baseline_era_driver_podium(train: pd.DataFrame, test: pd.DataFrame):
    """
    Era-aware driver podium rate: p(podium | driver, era(year))
    """
    train = train.copy()
    test = test.copy()

    train["era"] = train["year"].apply(get_era)
    test["era"] = test["year"].apply(get_era)

    grouped = train.groupby(["driverId", "era"])["is_podium"]
    cnt = grouped.count()
    pod = grouped.sum()
    stats = pd.DataFrame({"count": cnt, "podium": pod})
    stats["rate"] = stats["podium"] / stats["count"]
    stats = stats.reset_index()

    de_rate = stats.set_index(["driverId", "era"])["rate"].to_dict()

    # driver-global fallback
    g_driver = train.groupby("driverId")["is_podium"]
    d_cnt = g_driver.count()
    d_pod = g_driver.sum()
    p_driver = (d_pod / d_cnt).to_dict()

    p_global = train["is_podium"].mean()

    def score_row(row):
        key = (row["driverId"], row["era"])
        if key in de_rate:
            return de_rate[key]
        d = row["driverId"]
        if d in p_driver:
            return p_driver[d]
        return p_global

    test["score_era_driver"] = test.apply(score_row, axis=1)

    r3, top1 = race_level_metrics(test, "score_era_driver")
    print("\n=== Baseline: Era-aware driver podium rate ===")
    print(f"Global podium probability : {p_global:.4f}")
    print(f"Num (driver, era) pairs   : {len(de_rate)}")
    print(f"Recall@3 (podium hit rate): {r3:.4f}")
    print(f"Top-1 champion accuracy   : {top1:.4f}")


# ------------ NEW: Era × Driver × Weather baseline ------------

def baseline_era_driver_weather_podium(train: pd.DataFrame, test: pd.DataFrame):
    """
    Weather-aware baseline:
      p(podium | driver, era, weather_bucket)
    with fallback:
      (driver, era, weather) -> (driver, era) -> driver -> global
    """
    train = train.copy()
    test = test.copy()

    # era
    train["era"] = train["year"].apply(get_era)
    test["era"] = test["year"].apply(get_era)

    # weather bucket from precipitation
    for df in (train, test):
        df["weather_bucket"] = df["precipitation"].apply(bucket_precip)

    grouped = train.groupby(["driverId", "era", "weather_bucket"])["is_podium"]
    cnt = grouped.count()
    pod = grouped.sum()
    stats = pd.DataFrame({"count": cnt, "podium": pod})
    stats["rate"] = stats["podium"] / stats["count"]
    stats = stats.reset_index()

    dew_rate = stats.set_index(["driverId", "era", "weather_bucket"])["rate"].to_dict()

    # fallback: driver+era
    de_rate = train.groupby(["driverId", "era"])["is_podium"].mean().to_dict()
    # fallback: driver
    d_rate = train.groupby("driverId")["is_podium"].mean().to_dict()
    # global
    p_global = train["is_podium"].mean()

    def score_row(row):
        key3 = (row["driverId"], row["era"], row["weather_bucket"])
        if key3 in dew_rate:
            return dew_rate[key3]
        key2 = (row["driverId"], row["era"])
        if key2 in de_rate:
            return de_rate[key2]
        d = row["driverId"]
        if d in d_rate:
            return d_rate[d]
        return p_global

    test["score_era_driver_weather"] = test.apply(score_row, axis=1)

    r3, top1 = race_level_metrics(test, "score_era_driver_weather")
    print("\n=== Baseline: Era × Driver × Weather podium rate ===")
    print(f"Global podium probability             : {p_global:.4f}")
    print(f"Num (driver, era, weather) combinations: {len(dew_rate)}")
    print(f"Recall@3 (podium hit rate)           : {r3:.4f}")
    print(f"Top-1 champion accuracy              : {top1:.4f}")


def main():
    df, project_root = load_task1_df()

    # basic sanity: drop rows without race or driver or circuit
    df = df.dropna(subset=["raceId", "driverId", "circuitId"])

    # merge race-level weather
    race_weather = load_race_weather_map(project_root)
    df = df.merge(race_weather, on="raceId", how="left")

    train, val, test = split_by_year(df)

    # 1) Grid baseline
    baseline_grid(test)

    # 2) Global constant podium probability
    baseline_global_podium(train, test)

    # 3) Driver podium rate
    baseline_driver_podium(train, test)

    # 4) Driver–Circuit podium rate
    baseline_driver_circuit_podium(train, test, min_count=3)

    # 5) Star-driver baseline
    baseline_star_driver(train, test)

    # 6) Constructor podium rate
    baseline_constructor_podium(train, test)

    # 7) Circuit-only baseline
    baseline_circuit_podium(train, test)

    # 8) Era-aware driver podium baseline
    baseline_era_driver_podium(train, test)

    # 9) NEW: Era × Driver × Weather baseline
    baseline_era_driver_weather_podium(train, test)


if __name__ == "__main__":
    main()
