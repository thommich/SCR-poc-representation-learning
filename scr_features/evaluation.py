"""Matched-population LightGBM experiment ladder and paired OOT evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


KEY = ["application_id", "product"]
META = {"application_id", "product", "person_id", "application_time", "split",
        "target_value", "target_provenance", "customer_status", "coverage_status",
        "pit_basis", "authorization_status", "expected_split", "spine_duplicate_count"}


def _merge(base: pd.DataFrame, path: str, columns: list[str], name: str) -> pd.DataFrame:
    addon = pd.read_parquet(path)
    missing = set(KEY + columns) - set(addon)
    if missing:
        raise ValueError(f"{name} lacks {sorted(missing)}")
    if addon[KEY].duplicated().any():
        raise ValueError(f"{name} has duplicate application keys")
    merged = base.merge(addon[KEY + columns], on=KEY, how="left", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all() or len(merged) != len(base):
        raise ValueError(f"{name} does not match the frozen audience")
    return merged.drop(columns="_merge")


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"n": len(y), "auc": None, "gini": None, "ks": None, "brier": None}
    auc = roc_auc_score(y, prediction)
    fpr, tpr, _ = roc_curve(y, prediction)
    return {"n": len(y), "auc": float(auc), "gini": float(2 * auc - 1),
            "ks": float(np.max(tpr - fpr)), "brier": float(brier_score_loss(y, prediction))}


def _paired_ci(y, p0, p1, persons, months, *, repeats=200, seed=17) -> list[float] | None:
    if len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(seed)
    # Factorize once: repeated applications for a person within an OOT month
    # stay together. Avoid a per-person scan of the full audience.
    cluster = pd.MultiIndex.from_arrays([months, persons])
    codes, unique_clusters = pd.factorize(cluster, sort=False)
    order = np.argsort(codes, kind="stable")
    lengths = np.bincount(codes, minlength=len(unique_clusters))
    starts = np.r_[0, np.cumsum(lengths[:-1])]
    cluster_months = np.asarray(unique_clusters.get_level_values(0))
    month_pools = [np.flatnonzero(cluster_months == month)
                   for month in np.unique(cluster_months)]
    deltas = []
    for _ in range(repeats):
        sampled_clusters = np.concatenate([
            pool[rng.integers(0, len(pool), size=len(pool))] for pool in month_pools
        ])
        sampled_lengths = lengths[sampled_clusters]
        expanded = np.repeat(sampled_clusters, sampled_lengths)
        local_position = np.arange(len(expanded)) - np.repeat(
            np.cumsum(sampled_lengths) - sampled_lengths, sampled_lengths)
        sample = order[starts[expanded] + local_position]
        if len(np.unique(y[sample])) == 2:
            deltas.append(2 * (roc_auc_score(y[sample], p1[sample]) - roc_auc_score(y[sample], p0[sample])))
    return [float(v) for v in np.quantile(deltas, [0.025, 0.975])] if len(deltas) >= repeats // 2 else None


def _feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if missing := set(columns) - set(frame):
        raise ValueError(f"Missing model features: {sorted(missing)}")
    x = frame[columns].copy()
    for column in x:
        if x[column].dtype == object or str(x[column].dtype) == "string":
            x[column] = x[column].astype("category")
    return x


def temporal_probe(incumbent_path: str, app_embeddings: dict[str, str],
                   baseline_columns: list[str], *, product: str,
                   validation_month: str, train_provenances: list[str],
                   model_params: dict | None = None) -> dict:
    """Choose a temporal variant on train-period validation only."""
    base = pd.read_parquet(incumbent_path)
    base = base.loc[base["product"].eq(product)].reset_index(drop=True)
    if base.empty or base[KEY].duplicated().any():
        raise ValueError("Invalid incumbent population")
    base["application_month"] = pd.to_datetime(base["application_time"]).dt.to_period("M").astype(str)
    dev = (base["split"].eq("TRAIN") & base["application_month"].lt(validation_month) &
           base["target_provenance"].isin(train_provenances) & base["target_value"].notna())
    val = (base["split"].eq("TRAIN") & base["application_month"].eq(validation_month) &
           base["target_provenance"].isin(train_provenances) & base["target_value"].notna())
    if not dev.any() or not val.any():
        raise ValueError("Temporal probe requires development and validation rows")
    params = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
              "min_child_samples": 40, "random_state": 17, "n_jobs": 4, "verbosity": -1}
    if model_params:
        params.update(model_params)
    baseline_x = _feature_frame(base, baseline_columns)
    baseline_model = lgb.LGBMClassifier(**params)
    baseline_model.fit(baseline_x.loc[dev], base.loc[dev, "target_value"])
    baseline_pred = baseline_model.predict_proba(baseline_x.loc[val])[:, 1]
    baseline_metric = _metric(base.loc[val, "target_value"].to_numpy(), baseline_pred)
    report = {}
    for variant, path in app_embeddings.items():
        app = pd.read_parquet(path)
        hcols = sorted(c for c in app if c.startswith("h_") and c[2:].isdigit())
        joined = _merge(base, path, hcols, f"{variant} embeddings")
        features = baseline_columns + hcols
        x = _feature_frame(joined, features)
        model = lgb.LGBMClassifier(**params)
        model.fit(x.loc[dev], joined.loc[dev, "target_value"])
        pred = model.predict_proba(x.loc[val])[:, 1]
        report[variant] = _metric(joined.loc[val, "target_value"].to_numpy(), pred)
        report[variant]["missing_embedding_rate"] = float(joined[hcols].isna().all(axis=1).mean())
        if report[variant]["gini"] is not None and baseline_metric["gini"] is not None:
            report[variant]["delta_gini_vs_baseline"] = report[variant]["gini"] - baseline_metric["gini"]
    return {"product": product, "validation_month": validation_month,
            "baseline": baseline_metric, "variants": report}


def run_ladder(incumbent_path: str, conventional_path: str, app_path: str,
               geometry_path: str, geometry_manifest_path: str, out_dir: str,
               baseline_columns: list[str], *, product: str,
               train_provenances: list[str], validation_month: str,
               rolling_path: str | None = None, n_bootstrap: int = 200,
               model_params: dict | None = None,
               decision_threshold: float | None = None,
               minimum_delta_gini: float | None = None) -> dict:
    if not baseline_columns or any(c in META for c in baseline_columns):
        raise ValueError("baseline_columns must name only incumbent predictors")
    base = pd.read_parquet(incumbent_path)
    base = base.loc[base["product"].eq(product)].copy()
    if base.empty or base[KEY].duplicated().any():
        raise ValueError("Empty or duplicate incumbent application population")
    required = set(KEY) | {"person_id", "application_time", "split", "target_value", "target_provenance"}
    if required - set(base):
        raise ValueError("Incumbent file lacks evaluation metadata")
    conventional = pd.read_parquet(conventional_path)
    conventional_columns = [c for c in conventional if c not in KEY]
    base = _merge(base, conventional_path, conventional_columns, "conventional SCR")
    app = pd.read_parquet(app_path)
    raw_columns = sorted(c for c in app if c.startswith("h_") and c[2:].isdigit())
    extra_meta = [c for c in ("coverage_status", "customer_status", "pit_basis") if c not in base]
    base = _merge(base, app_path, raw_columns + extra_meta, "application embeddings")
    geom_manifest = json.loads(Path(geometry_manifest_path).read_text(encoding="utf-8"))
    blocks = geom_manifest["feature_groups"]
    geometry_columns = blocks["static_geometry"] + blocks["reference_trajectory"]
    base = _merge(base, geometry_path, geometry_columns, "geometry")
    rolling_columns = []
    if rolling_path:
        rolling = pd.read_parquet(rolling_path)
        rolling_columns = [c for c in rolling if c.startswith("scr_rolling_h_")]
        base = _merge(base, rolling_path, rolling_columns, "rolling geometry")
    base["application_time"] = pd.to_datetime(base["application_time"])
    base["application_month"] = base["application_time"].dt.to_period("M").astype(str)
    if not base["split"].isin(["TRAIN", "OOT"]).all():
        raise ValueError("Unexpected split in incumbent population")
    valid_train = base["split"].eq("TRAIN") & base["target_provenance"].isin(train_provenances) & base["target_value"].notna()
    development = valid_train & base["application_month"].lt(validation_month)
    validation = valid_train & base["application_month"].eq(validation_month)
    oot = base["split"].eq("OOT") & base["target_provenance"].eq("OBSERVED") & base["target_value"].notna()
    if min(development.sum(), validation.sum(), oot.sum()) == 0:
        raise ValueError("Empty development, validation or observed OOT cohort")
    if not base.loc[valid_train | oot, "target_value"].isin([0, 1]).all():
        raise ValueError("Target must be an already constructed binary product-specific label")
    designs = {
        "B0": baseline_columns,
        "B1": baseline_columns + conventional_columns,
        "B2": baseline_columns + blocks["static_geometry"],
        "B3": baseline_columns + conventional_columns + blocks["static_geometry"],
        "B4": baseline_columns + conventional_columns + blocks["static_geometry"] + blocks["reference_trajectory"],
        "B6": baseline_columns + conventional_columns + raw_columns,
    }
    if rolling_columns:
        designs["B5"] = baseline_columns + conventional_columns + blocks["static_geometry"] + rolling_columns
    params = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
              "min_child_samples": 40, "random_state": 17, "n_jobs": 4, "verbosity": -1}
    if model_params:
        params.update(model_params)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = {}
    predictions = {}
    for name, columns in designs.items():
        if len(columns) != len(set(columns)):
            raise ValueError(f"Repeated feature in {name}")
        x = _feature_frame(base, columns)
        model = lgb.LGBMClassifier(**params)
        training_started = time.perf_counter()
        model.fit(x.loc[development], base.loc[development, "target_value"],
                  eval_set=[(x.loc[validation], base.loc[validation, "target_value"])],
                  eval_metric="auc",
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        best = model.best_iteration_ or params["n_estimators"]
        validation_pred = model.predict_proba(x.loc[validation])[:, 1]
        final = lgb.LGBMClassifier(**{**params, "n_estimators": best})
        final.fit(x.loc[valid_train], base.loc[valid_train, "target_value"])
        training_seconds = time.perf_counter() - training_started
        scoring_started = time.perf_counter()
        oot_pred = final.predict_proba(x.loc[oot])[:, 1]
        scoring_seconds = time.perf_counter() - scoring_started
        predictions[name] = oot_pred
        scores[name] = {"validation": _metric(base.loc[validation, "target_value"].to_numpy(), validation_pred),
                        "oot_observed": _metric(base.loc[oot, "target_value"].to_numpy(), oot_pred),
                        "best_iteration": best, "n_features": len(columns),
                        "training_seconds": training_seconds,
                        "oot_scoring_seconds": scoring_seconds,
                        "feature_columns": columns}
        final.booster_.save_model(str(out / f"{name}.txt"))
    observed = base.loc[oot, KEY + ["person_id", "application_month", "target_value"]].copy()
    for column in ("coverage_status", "customer_status"):
        if column in base:
            observed[column] = base.loc[oot, column].to_numpy()
    for name, values in predictions.items():
        observed[f"score_{name}"] = values
    observed.to_parquet(out / "oot_predictions.parquet", index=False)
    y = observed["target_value"].to_numpy()
    people = observed["person_id"].astype(str).to_numpy()
    months = observed["application_month"].to_numpy()
    for name in scores:
        scores[name]["oot_by_month"] = {
            month: _metric(group["target_value"].to_numpy(), group[f"score_{name}"].to_numpy())
            for month, group in observed.groupby("application_month")
        }
        for segment in ("coverage_status", "customer_status"):
            if segment in observed:
                scores[name][f"oot_by_{segment}"] = {
                    str(value): _metric(group["target_value"].to_numpy(), group[f"score_{name}"].to_numpy())
                    for value, group in observed.groupby(segment, dropna=False)
                }
        if name != "B0":
            reference = "B3" if name in {"B4", "B5"} else ("B1" if name in {"B3", "B6"} else "B0")
            scores[name]["paired_delta_gini_vs"] = reference
            scores[name]["paired_delta_gini_ci"] = _paired_ci(y, predictions[reference], predictions[name],
                                                                 people, months, repeats=n_bootstrap)
            if minimum_delta_gini is not None:
                point_delta = scores[name]["oot_observed"]["gini"] - scores[reference]["oot_observed"]["gini"]
                interval = scores[name]["paired_delta_gini_ci"]
                scores[name]["statistical_economic_screen"] = bool(
                    interval is not None and interval[0] > 0 and point_delta >= minimum_delta_gini)
            base_band = np.minimum(9, (observed["score_B0"].rank(pct=True).to_numpy() * 10).astype(int))
            new_band = np.minimum(9, (observed[f"score_{name}"].rank(pct=True).to_numpy() * 10).astype(int))
            scores[name]["oot_decile_movement_rate_vs_B0"] = float(np.mean(base_band != new_band))
            if decision_threshold is not None:
                scores[name]["oot_threshold_crossing_rate_vs_B0"] = float(np.mean(
                    (predictions["B0"] >= decision_threshold) != (predictions[name] >= decision_threshold)))
    report = {"product": product, "population": len(base), "development": int(development.sum()),
              "validation": int(validation.sum()), "oot_observed": int(oot.sum()),
              "train_provenances": train_provenances, "validation_month": validation_month,
              "minimum_delta_gini": minimum_delta_gini, "decision_threshold": decision_threshold,
              "models": scores}
    (out / "scorecard.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
