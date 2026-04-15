import pandas as pd
from pathlib import Path

# We will only keep races from this year onwards
MIN_YEAR = 2000


def load_core_tables():
    """
    Load main CSV tables from data_raw/f1_core.
    Paths are resolved relative to this script file.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_core = project_root / "data_raw" / "f1_core"
    data_processed = project_root / "data_processed"
    data_processed.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Loading core CSVs from: {data_core}")

    races = pd.read_csv(data_core / "races.csv")
    results = pd.read_csv(data_core / "results.csv")
    drivers = pd.read_csv(data_core / "drivers.csv")
    constructors = pd.read_csv(data_core / "constructors.csv")
    driver_standings = pd.read_csv(data_core / "driver_standings.csv")

    return project_root, data_processed, races, results, drivers, constructors, driver_standings


def filter_years(races, results, driver_standings, min_year=MIN_YEAR):
    """
    Keep only races from min_year onwards and filter results/standings accordingly.
    """
    races_f = races[races["year"] >= min_year].copy()
    valid_race_ids = races_f["raceId"].unique()

    results_f = results[results["raceId"].isin(valid_race_ids)].copy()
    ds_f = driver_standings[driver_standings["raceId"].isin(valid_race_ids)].copy()

    return races_f, results_f, ds_f


def build_task1(races, results, drivers, constructors, driver_standings, out_dir: Path):
    """
    Build Task 1 dataset:
    - One row per (raceId, driverId)
    - Features: year, round, circuitId, driver/constructor info,
                pre-race points/rank, rolling history, etc.
    - Labels:
        * is_podium  (1 if finish_pos in [1, 2, 3])
        * finish_pos (numeric finishing position)
    """
    print("Building Task 1 (race-driver podium prediction) dataset...")

    # Merge basic info: race-level to results
    res = results.merge(
        races[["raceId", "year", "round", "circuitId"]],
        on="raceId",
        how="left",
    )

    # Merge driver info
    res = res.merge(
        drivers[["driverId", "driverRef", "code", "forename", "surname", "nationality"]],
        on="driverId",
        how="left",
    )

    # Merge constructor info
    res = res.merge(
        constructors[["constructorId", "name", "nationality"]].rename(
            columns={"name": "constructor_name", "nationality": "constructor_nationality"}
        ),
        on="constructorId",
        how="left",
    )

    # Finishing position column
    if "positionOrder" in res.columns:
        res["finish_pos"] = res["positionOrder"]
    else:
        # Fallback: try to convert 'position' string to numeric
        res["finish_pos"] = pd.to_numeric(res.get("position"), errors="coerce")

    # Podium label: 1 if final position in [1, 2, 3]
    res["is_podium"] = ((res["finish_pos"] >= 1) & (res["finish_pos"] <= 3)).astype(int)

    # ==== Pre-race points and rank from driver_standings ====
    ds = driver_standings.merge(
        races[["raceId", "year", "round"]],
        on="raceId",
        how="left",
    )

    ds = ds.sort_values(["year", "driverId", "round"])

    # post-race points/rank -> shift by 1 race to get "pre-race"
    ds["prerace_points"] = ds.groupby(["year", "driverId"])["points"].shift(1)
    ds["prerace_rank"] = ds.groupby(["year", "driverId"])["position"].shift(1)

    ds["prerace_points"] = ds["prerace_points"].fillna(0.0)
    max_rank = ds["position"].max()
    ds["prerace_rank"] = ds["prerace_rank"].fillna(max_rank)

    res = res.merge(
        ds[["raceId", "driverId", "prerace_points", "prerace_rank"]],
        on=["raceId", "driverId"],
        how="left",
    )

    res["prerace_points"] = res["prerace_points"].fillna(0.0)
    res["prerace_rank"] = res["prerace_rank"].fillna(max_rank)

        # ==== Rolling history features (last 3 races) ====
    # We use groupby().transform(...) to keep the same index as 'res'
    res = res.sort_values(["driverId", "year", "round"])
    group = res.groupby("driverId", group_keys=False)

    # previous race finishing position
    res["prev1_finish"] = group["finish_pos"].shift(1)

    # rolling average finish over last 3 races (excluding current)
    res["prev3_avg_finish"] = group["finish_pos"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean().shift(1)
    )

    # rolling sum of points over last 3 races (excluding current)
    res["prev3_points"] = group["points"].transform(
        lambda s: s.rolling(window=3, min_periods=1).sum().shift(1)
    )

    # Fill NaNs for the first few races per driver
    for col in ["prev1_finish", "prev3_avg_finish", "prev3_points"]:
        res[col] = res[col].fillna(res[col].median())

    out_path = out_dir / "task1_race_driver_podium.csv"
    res.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Task 1 dataset saved to: {out_path}")


def build_task2(races, results, driver_standings, out_dir: Path):
    """
    Build Task 2 dataset:
    - One row per (year, driverId)
    - Features: mid-season aggregates (points, average finish, wins, podiums, races)
                + previous season final points/rank
    - Labels:
        * final_points
        * final_rank
    """
    print("Building Task 2 (season final points prediction) dataset...")

    # === Final season points/rank ===
    ds = driver_standings.merge(
        races[["raceId", "year", "round"]],
        on="raceId",
        how="left",
    )

    ds = ds.sort_values(["year", "driverId", "round"])
    ds_final = ds.groupby(["year", "driverId"], as_index=False).tail(1)

    ds_final = ds_final[["year", "driverId", "points", "position"]].rename(
        columns={"points": "final_points", "position": "final_rank"}
    )

    # === Mid-season aggregates (up to mid_round) ===
    season_rounds = races.groupby("year")["round"].max().rename("max_round")
    races_season = races.merge(season_rounds, on="year", how="left")
    races_season["mid_round"] = (races_season["max_round"] // 2)

    res2 = results.merge(
        races_season[["raceId", "year", "round", "mid_round"]],
        on="raceId",
        how="left",
    )

    mid_res = res2[res2["round"] <= res2["mid_round"]].copy()

    # Determine which column to use for finishing position
    if "positionOrder" in mid_res.columns:
        finish_col = "positionOrder"
    else:
        mid_res["position_numeric"] = pd.to_numeric(
            mid_res.get("position"), errors="coerce"
        )
        finish_col = "position_numeric"

    agg_mid = mid_res.groupby(["year", "driverId"]).agg(
        mid_points=("points", "sum"),
        mid_avg_finish=(finish_col, "mean"),
        mid_races=("raceId", "nunique"),
        mid_wins=(finish_col, lambda x: (x == 1).sum()),
        mid_podiums=(finish_col, lambda x: (x <= 3).sum()),
    ).reset_index()

    # === Previous season info (year-1 final) ===
    prev_season = ds_final.copy()
    prev_season["year"] = prev_season["year"] + 1
    prev_season = prev_season.rename(
        columns={
            "final_points": "prev_final_points",
            "final_rank": "prev_final_rank",
        }
    )

    agg_mid = agg_mid.merge(
        prev_season[["year", "driverId", "prev_final_points", "prev_final_rank"]],
        on=["year", "driverId"],
        how="left",
    )

    agg_mid["prev_final_points"] = agg_mid["prev_final_points"].fillna(0.0)
    max_prev_rank = agg_mid["prev_final_rank"].max()
    agg_mid["prev_final_rank"] = agg_mid["prev_final_rank"].fillna(max_prev_rank)

    # === Merge with final labels ===
    task2_df = agg_mid.merge(ds_final, on=["year", "driverId"], how="inner")

    out_path = out_dir / "task2_driver_season_points_midround.csv"
    task2_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Task 2 dataset saved to: {out_path}")


def main():
    (
        project_root,
        data_processed,
        races,
        results,
        drivers,
        constructors,
        driver_standings,
    ) = load_core_tables()

    races_f, results_f, ds_f = filter_years(races, results, driver_standings, MIN_YEAR)

    print(f"Kept races with year >= {MIN_YEAR}: {len(races_f)} rows")
    print(f"Kept results for those races      : {len(results_f)} rows")
    print(f"Kept driver_standings rows        : {len(ds_f)} rows")

    build_task1(races_f, results_f, drivers, constructors, ds_f, data_processed)
    build_task2(races_f, results_f, ds_f, data_processed)


if __name__ == "__main__":
    main()
