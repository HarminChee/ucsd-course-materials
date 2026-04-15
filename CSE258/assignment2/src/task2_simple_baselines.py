import numpy as np
import pandas as pd
from pathlib import Path

MIN_YEAR = 2000


def load_task2_df():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data_processed" / "task2_driver_season_points_midround.csv"
    print(f"Loading Task2 data from: {data_path}")
    df = pd.read_csv(data_path)
    df["year"] = df["year"].astype(int)
    return df, project_root


def split_by_year(df: pd.DataFrame):
    train_years = list(range(2000, 2015))
    val_years = list(range(2015, 2019))
    test_years = list(range(2019, 2024))

    train = df[df["year"].isin(train_years)].copy()
    val = df[df["year"].isin(val_years)].copy()
    test = df[df["year"].isin(test_years)].copy()

    print(f"Train seasons: {len(train)}")
    print(f"Val   seasons: {len(val)}")
    print(f"Test  seasons: {len(test)}")

    return train, val, test


def regression_metrics(y_true, y_pred, name: str):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    var_y = np.var(y_true)
    if var_y > 0:
        r2 = 1.0 - np.mean((y_true - y_pred) ** 2) / var_y
    else:
        r2 = np.nan

    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")

    print(f"\n=== {name} ===")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R^2      : {r2:.4f}")
    print(f"Spearman : {spearman:.4f}")


# ---------- 简单均值类 baseline ----------

def baseline_global_mean(train: pd.DataFrame, test: pd.DataFrame):
    global_mean = train["final_points"].mean()
    y_true = test["final_points"].values
    y_pred = np.full_like(y_true, fill_value=global_mean, dtype=float)
    regression_metrics(y_true, y_pred, name="Baseline: Global mean final_points")


def baseline_driver_mean(train: pd.DataFrame, test: pd.DataFrame):
    grouped = train.groupby("driverId")["final_points"]
    driver_mean = grouped.mean().to_dict()
    global_mean = train["final_points"].mean()

    def predict_row(row):
        d = row["driverId"]
        return driver_mean.get(d, global_mean)

    y_true = test["final_points"].values
    y_pred = test.apply(predict_row, axis=1).values
    regression_metrics(y_true, y_pred, name="Baseline: Driver historical mean final_points")


def baseline_mid_points(test: pd.DataFrame):
    if "mid_points" not in test.columns:
        print("\n[Warning] mid_points not found; skipping mid_points baseline.")
        return
    y_true = test["final_points"].values
    y_pred = test["mid_points"].values
    regression_metrics(y_true, y_pred, name="Baseline: mid_points as final_points")


# ---------- Era 定义 ----------

def get_era(year: int) -> str:
    if year <= 2013:
        return "pre_hybrid"
    elif year <= 2021:
        return "v6_hybrid"
    else:
        return "ground_effect"


def baseline_era_driver_mean(train: pd.DataFrame, test: pd.DataFrame):
    """
    Era-aware driver baseline:
      E[final_points | driver, era(year)] with fallback to
      driver-global mean and global mean.
    """
    train = train.copy()
    test = test.copy()

    train["era"] = train["year"].apply(get_era)
    test["era"] = test["year"].apply(get_era)

    grouped = train.groupby(["driverId", "era"])["final_points"]
    era_mean = grouped.mean().to_dict()

    # driver-global fallback
    driver_mean = train.groupby("driverId")["final_points"].mean().to_dict()
    global_mean = train["final_points"].mean()

    def predict_row(row):
        key = (row["driverId"], row["era"])
        if key in era_mean:
            return era_mean[key]
        d = row["driverId"]
        if d in driver_mean:
            return driver_mean[d]
        return global_mean

    y_true = test["final_points"].values
    y_pred = test.apply(predict_row, axis=1).values
    regression_metrics(y_true, y_pred, name="Baseline: Era-aware driver mean final_points")


# ---------- 从原始 races/results 读车队信息 ----------

def load_core_results(project_root: Path) -> pd.DataFrame:
    """
    Load original races/results to derive (year, driverId, constructorId) info.
    """
    data_core = project_root / "data_raw" / "f1_core"
    races = pd.read_csv(data_core / "races.csv")
    results = pd.read_csv(data_core / "results.csv")

    races = races[races["year"] >= MIN_YEAR].copy()
    valid_race_ids = races["raceId"].unique()
    results = results[results["raceId"].isin(valid_race_ids)].copy()

    res = results.merge(
        races[["raceId", "year"]],
        on="raceId",
        how="left",
    )
    return res


def build_driver_season_constructor_map(project_root: Path):
    """
    For each (year, driverId), choose the constructorId that appears
    most frequently in that season as the main team.
    Returns: dict[(year, driverId)] -> constructorId
    """
    res = load_core_results(project_root)

    grouped = res.groupby(["year", "driverId", "constructorId"])["raceId"].count()
    temp = grouped.reset_index().rename(columns={"raceId": "race_count"})

    # for each (year, driverId), pick constructorId with max race_count
    idx = temp.groupby(["year", "driverId"])["race_count"].idxmax()
    main_team = temp.loc[idx, ["year", "driverId", "constructorId"]]

    mapping = main_team.set_index(["year", "driverId"])["constructorId"].to_dict()
    return mapping


def baseline_constructor_mean(train: pd.DataFrame, test: pd.DataFrame, project_root: Path):
    """
    Constructor baseline:
      E[final_points | constructorId] with fallback to global mean.
    """
    mapping = build_driver_season_constructor_map(project_root)

    def attach_constructor(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def get_cons(row):
            key = (row["year"], row["driverId"])
            return mapping.get(key, -1)

        df["constructorId"] = df.apply(get_cons, axis=1)
        return df

    train_c = attach_constructor(train)
    test_c = attach_constructor(test)

    grouped = train_c.groupby("constructorId")["final_points"]
    cons_mean = grouped.mean().to_dict()
    global_mean = train_c["final_points"].mean()

    def predict_row(row):
        c = row["constructorId"]
        return cons_mean.get(c, global_mean)

    y_true = test_c["final_points"].values
    y_pred = test_c.apply(predict_row, axis=1).values
    regression_metrics(y_true, y_pred, name="Baseline: Constructor mean final_points")


# ---------- Weather 相关 ----------

def bucket_precip(x: float) -> str:
    """
    将赛季平均降水分桶：dry / light_rain / heavy_rain / unknown
    """
    if pd.isna(x):
        return "unknown"
    if x <= 0.0:
        return "dry"
    if x < 1.0:
        return "light_rain"
    return "heavy_rain"


def load_race_weather_map(project_root: Path) -> pd.DataFrame:
    """
    raceId -> (temperature, precipitation, windspeed)
    """
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


def build_driver_season_weather_bucket_map(project_root: Path):
    """
    对每个 (year, driverId) 统计该车手当赛季参赛所有分站的平均降水，
    然后分桶为 season_weather_bucket。
    返回: dict[(year, driverId)] -> bucket(str)
    """
    res = load_core_results(project_root)
    race_weather = load_race_weather_map(project_root)

    res_w = res.merge(race_weather, on="raceId", how="left")

    grouped = res_w.groupby(["year", "driverId"])["precipitation"]
    avg_precip = grouped.mean().reset_index().rename(columns={"precipitation": "avg_precip"})

    avg_precip["season_weather_bucket"] = avg_precip["avg_precip"].apply(bucket_precip)

    mapping = avg_precip.set_index(["year", "driverId"])["season_weather_bucket"].to_dict()
    return mapping


def baseline_constructor_weather_mean(train: pd.DataFrame, test: pd.DataFrame, project_root: Path):
    """
    Weather-aware constructor baseline:
      E[final_points | constructorId, season_weather_bucket]
    fallback:
      (constructor, bucket) -> constructor -> global
    """
    cons_map = build_driver_season_constructor_map(project_root)
    weather_bucket_map = build_driver_season_weather_bucket_map(project_root)

    def attach(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def get_cons(row):
            key = (row["year"], row["driverId"])
            return cons_map.get(key, -1)

        def get_bucket(row):
            key = (row["year"], row["driverId"])
            return weather_bucket_map.get(key, "unknown")

        df["constructorId"] = df.apply(get_cons, axis=1)
        df["season_weather_bucket"] = df.apply(get_bucket, axis=1)
        return df

    train_c = attach(train)
    test_c = attach(test)

    grouped = train_c.groupby(["constructorId", "season_weather_bucket"])["final_points"]
    cons_w_mean = grouped.mean().to_dict()

    cons_mean = train_c.groupby("constructorId")["final_points"].mean().to_dict()
    global_mean = train_c["final_points"].mean()

    def predict_row(row):
        key = (row["constructorId"], row["season_weather_bucket"])
        if key in cons_w_mean:
            return cons_w_mean[key]
        c = row["constructorId"]
        if c in cons_mean:
            return cons_mean[c]
        return global_mean

    y_true = test_c["final_points"].values
    y_pred = test_c.apply(predict_row, axis=1).values
    regression_metrics(y_true, y_pred, name="Baseline: Constructor × Weather mean final_points")


def main():
    df, project_root = load_task2_df()
    df = df.dropna(subset=["year", "driverId", "final_points"])

    train, val, test = split_by_year(df)

    # 1) Global mean baseline
    baseline_global_mean(train, test)

    # 2) Driver historical mean baseline
    baseline_driver_mean(train, test)

    # 3) mid_points baseline
    baseline_mid_points(test)

    # 4) Era-aware driver mean baseline
    baseline_era_driver_mean(train, test)

    # 5) Constructor baseline
    baseline_constructor_mean(train, test, project_root)

    # 6) NEW: Constructor × Weather baseline
    baseline_constructor_weather_mean(train, test, project_root)


if __name__ == "__main__":
    main()
