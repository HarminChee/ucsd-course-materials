import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


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


# ---------- core + weather 复用 ----------

def load_core_results(project_root: Path) -> pd.DataFrame:
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


def build_driver_season_constructor_map(project_root: Path):
    res = load_core_results(project_root)

    grouped = res.groupby(["year", "driverId", "constructorId"])["raceId"].count()
    temp = grouped.reset_index().rename(columns={"raceId": "race_count"})

    idx = temp.groupby(["year", "driverId"])["race_count"].idxmax()
    main_team = temp.loc[idx, ["year", "driverId", "constructorId"]]

    mapping = main_team.set_index(["year", "driverId"])["constructorId"].to_dict()
    return mapping


def build_driver_season_weather_numeric_map(project_root: Path):
    """
    对每个 (year, driverId) 统计赛季平均 temperature / precipitation / windspeed
    返回: dict[(year, driverId)] -> dict{temperature, precipitation, windspeed}
    """
    res = load_core_results(project_root)
    race_weather = load_race_weather_map(project_root)

    res_w = res.merge(race_weather, on="raceId", how="left")

    grouped = res_w.groupby(["year", "driverId"])[["temperature", "precipitation", "windspeed"]]
    agg = grouped.mean().reset_index().rename(
        columns={
            "temperature": "temp_avg",
            "precipitation": "precip_avg",
            "windspeed": "wind_avg",
        }
    )

    mapping = agg.set_index(["year", "driverId"])[["temp_avg", "precip_avg", "wind_avg"]].to_dict("index")
    return mapping


def main():
    df, project_root = load_task2_df()
    df = df.dropna(subset=["year", "driverId", "final_points"])

    # attach constructorId and season-average weather
    cons_map = build_driver_season_constructor_map(project_root)
    weather_map = build_driver_season_weather_numeric_map(project_root)

    def enrich(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()

        def get_cons(row):
            key = (row["year"], row["driverId"])
            return cons_map.get(key, -1)

        def get_weather(row, key_name):
            key = (row["year"], row["driverId"])
            info = weather_map.get(key, None)
            if info is None:
                return np.nan
            return info[key_name]

        df["constructorId"] = df.apply(get_cons, axis=1)
        df["temp_avg"] = df.apply(lambda r: get_weather(r, "temp_avg"), axis=1)
        df["precip_avg"] = df.apply(lambda r: get_weather(r, "precip_avg"), axis=1)
        df["wind_avg"] = df.apply(lambda r: get_weather(r, "wind_avg"), axis=1)

        return df

    df = enrich(df)

    train, val, test = split_by_year(df)

    # 特征选择：mid_points + year + 赛季平均天气 + constructorId
    feat_num = ["mid_points", "year", "temp_avg", "precip_avg", "wind_avg"]
    feat_cat = ["constructorId"]

    # 缺失值用 train 的中位数填充
    num_medians = train[feat_num].median()
    for split in (train, val, test):
        for col in feat_num:
            if col not in split.columns:
                split[col] = np.nan
        split[feat_num] = split[feat_num].fillna(num_medians)

    # 目标
    y_train = train["final_points"].values
    y_val = val["final_points"].values
    y_test = test["final_points"].values

    X_train = train[feat_num + feat_cat]
    X_val = val[feat_num + feat_cat]
    X_test = test[feat_num + feat_cat]

    # 预处理：数值特征 StandardScaler，类别特征 OneHot
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feat_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat),
        ]
    )

    model = LinearRegression()

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)

    # 在 test 集上评估
    y_pred_test = pipe.predict(X_test)
    regression_metrics(y_test, y_pred_test, name="Linear Regression baseline (mid_points + constructor + weather)")


if __name__ == "__main__":
    main()
