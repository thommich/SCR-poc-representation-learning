"""Summarize SCR records and fit/score a frozen representation challenger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

META = ["application_id", "person_id", "application_time", "product", "split", "expected_split", "spine_duplicate_count", "pit_basis", "authorization_status", "target_value", "target_provenance", "customer_status", "coverage_status", "report_months", "no_report_months", "no_exposure_months"]
MEASURES = ["balance_current", "balance_overdue", "record_count", "modality_count"]
ALLOWED_STATUS = {"USABLE", "INSUFFICIENT_HISTORY", "NO_AUTHORIZATION", "UNKNOWN_AVAILABILITY", "NO_REPORT", "NO_EXPOSURE"}


def _read(raw_glob: str):
    con = duckdb.connect()
    # The relation API passes the path without interpolating it into SQL.
    con.from_parquet(raw_glob).create_view("raw")
    columns = {row[1] for row in con.execute("PRAGMA table_info('raw')").fetchall()}
    required = set(META) | {"event_id", "scr_ref_month", "available_at", "snapshot_status", "record_key", "modality", "balance_current", "balance_overdue"}
    if missing := required - columns:
        raise ValueError(f"Extract is missing columns: {sorted(missing)}")
    return con


def _assert_raw(con):
    bad = con.execute("""
       SELECT COUNT(*) FROM raw
       WHERE application_id IS NULL OR person_id IS NULL OR application_time IS NULL
          OR product IS NULL OR split IS NULL OR target_provenance IS NULL
          OR expected_split IS NULL OR split <> expected_split
          OR spine_duplicate_count <> 1 OR pit_basis IS NULL
          OR authorization_status IS NULL OR coverage_status IS NULL
          OR coverage_status IN ('AMBIGUOUS_AUTHORIZATION', 'AMBIGUOUS_VERSION')
          OR (event_id IS NOT NULL AND (available_at IS NULL OR available_at >= application_time))
          OR (scr_ref_month IS NOT NULL AND
              (TRY_STRPTIME(scr_ref_month, '%Y%m') IS NULL OR
               TRY_STRPTIME(scr_ref_month, '%Y%m') >= DATE_TRUNC('month', application_time)))
          OR (snapshot_status = 'REPORT' AND event_id IS NULL)
          OR (record_key IS NOT NULL AND snapshot_status <> 'REPORT')
          OR (snapshot_status = 'REPORT' AND record_key IS NULL)
    """).fetchone()[0]
    if bad:
        raise ValueError(f"Point-in-time or source integrity violation in {bad} extract rows")
    bad_status = con.execute("SELECT DISTINCT coverage_status FROM raw").fetchall()
    if set(x[0] for x in bad_status) - ALLOWED_STATUS:
        raise ValueError("Unrecognized coverage status")
    n, n_unique = con.execute("""
       SELECT COUNT(*), COUNT(DISTINCT (application_id, product))
       FROM (SELECT DISTINCT application_id, product, person_id, application_time,
                             split, target_value, target_provenance, customer_status,
                             coverage_status, report_months, no_report_months, no_exposure_months
             FROM raw)
    """).fetchone()
    if n != n_unique:
        raise ValueError("Conflicting application metadata for one application key")
    duplicates = con.execute("""
      SELECT COUNT(*) FROM (
        SELECT application_id, product, event_id, scr_ref_month,
               record_key, modality, institution, origin, COUNT(*) n
        FROM raw WHERE record_key IS NOT NULL
        GROUP BY ALL HAVING COUNT(*) > 1)
    """).fetchone()[0]
    if duplicates:
        raise ValueError("Duplicated source records after snapshot join")


def summarize(raw_glob: str, history_months: int = 12) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if not 1 <= history_months <= 60:
        raise ValueError("history_months outside supported range")
    con = _read(raw_glob)
    _assert_raw(con)
    apps = con.execute(f"SELECT DISTINCT {', '.join(META)} FROM raw").df()
    if apps.empty:
        raise ValueError("Empty extract")
    apps["application_time"] = pd.to_datetime(apps["application_time"])
    if apps["application_id"].duplicated().any() and apps[["application_id", "product"]].duplicated().any():
        raise ValueError("Duplicate application key")
    monthly = con.execute("""
       SELECT application_id, product, scr_ref_month,
              SUM(COALESCE(balance_current, 0)) balance_current,
              SUM(COALESCE(balance_overdue, 0)) balance_overdue,
              COUNT(*) record_count, COUNT(DISTINCT modality) modality_count
       FROM raw WHERE snapshot_status = 'REPORT' AND record_key IS NOT NULL
       GROUP BY application_id, product, scr_ref_month
    """).df()
    con.close()
    names = [f"scr_m{lag:02d}_{name}" for lag in range(1, history_months + 1) for name in MEASURES]
    matrix = np.zeros((len(apps), len(names)), dtype=np.float32)
    app_index = pd.MultiIndex.from_frame(apps[["application_id", "product"]])
    if len(monthly):
        m_index = pd.MultiIndex.from_frame(monthly[["application_id", "product"]])
        row = app_index.get_indexer(m_index)
        if (row < 0).any():
            raise ValueError("Monthly records have no application")
        app_period = apps["application_time"].dt.to_period("M")
        ref_period = pd.PeriodIndex(monthly["scr_ref_month"].astype(str), freq="M")
        age = (app_period.iloc[row].dt.year.to_numpy() - ref_period.year) * 12 + (app_period.iloc[row].dt.month.to_numpy() - ref_period.month)
        if ((age < 1) | (age > history_months)).any():
            raise ValueError("SCR month outside configured trailing window")
        for j, name in enumerate(MEASURES):
            values = monthly[name].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite SCR amount or count")
            matrix[row, (age - 1) * len(MEASURES) + j] = values
    # Raw money gets a signed log transform; counts remain in their original scale.
    for j, name in enumerate(MEASURES):
        if name.startswith("balance"):
            sl = matrix[:, j::len(MEASURES)]
            matrix[:, j::len(MEASURES)] = np.sign(sl) * np.log1p(np.abs(sl))
    return apps, matrix, names


def _conventional(apps: pd.DataFrame, matrix: np.ndarray, history_months: int) -> pd.DataFrame:
    out = pd.DataFrame(index=apps.index)
    out["scr_report_months"] = apps["report_months"].astype("float32")
    for width in (3, 6, history_months):
        width = min(width, history_months)
        for j, measure in enumerate(MEASURES):
            values = matrix[:, j:width * len(MEASURES):len(MEASURES)]
            out[f"scr_{measure}_logsum_{width}m"] = values.sum(axis=1)
    due = matrix[:, 1::len(MEASURES)]
    out["scr_overdue_logmax"] = due.max(axis=1)
    out["scr_latest_overdue_log"] = due[:, 0]
    out["scr_months_with_records"] = (matrix[:, 2::len(MEASURES)] > 0).sum(axis=1)
    out["scr_last_record_lag"] = np.where((matrix[:, 2::len(MEASURES)] > 0).any(axis=1),
                                         (matrix[:, 2::len(MEASURES)] > 0).argmax(axis=1) + 1, np.nan)
    return out


def export_conventional(raw_glob: str, output: str, history_months: int = 12) -> dict:
    """Export B1 conventional SCR features without fitting any representation."""
    apps, matrix, _ = summarize(raw_glob, history_months)
    conventional = _conventional(apps, matrix, history_months)
    out = apps[["application_id", "product"]].copy()
    for column in conventional:
        out[column] = conventional[column]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return {"applications": len(out), "feature_columns": list(conventional.columns)}


def _embedding(model: dict, matrix: np.ndarray) -> np.ndarray:
    scaled = model["scaler"].transform(matrix)
    if model["encoder_type"] == "pca":
        return model["encoder"].transform(scaled)
    network = model["encoder"]
    first = np.maximum(0, scaled @ network.coefs_[0] + network.intercepts_[0])
    return np.maximum(0, first @ network.coefs_[1] + network.intercepts_[1])


def fit(raw_glob: str, earliest_application: str, out_dir: str, *, history_months: int = 12,
        dimensions: int = 16, clusters: int = 8, encoder_type: str = "autoencoder",
        max_fit_anchors: int = 100_000) -> dict:
    apps, matrix, names = summarize(raw_glob, history_months)
    if apps["pit_basis"].nunique() != 1:
        raise ValueError("Mixed point-in-time source adapters in fit corpus")
    earliest = pd.Timestamp(earliest_application)
    if apps["application_time"].max() >= earliest:
        raise ValueError("Encoder fit corpus is not strictly before the earliest application")
    usable = apps["coverage_status"].eq("USABLE").to_numpy()
    if encoder_type not in {"autoencoder", "pca"}:
        raise ValueError("encoder_type must be autoencoder or pca")
    if dimensions < 1 or clusters < 2:
        raise ValueError("dimensions must be positive and clusters must be at least two")
    if usable.sum() < max(dimensions + 1, clusters + 1) or dimensions > matrix.shape[1]:
        raise ValueError("Too few usable anchors or input features for the chosen encoder")
    if max_fit_anchors < max(dimensions + 1, clusters + 1):
        raise ValueError("max_fit_anchors is too small")
    candidates = np.flatnonzero(usable)
    rng = np.random.default_rng(17)
    chosen = rng.choice(candidates, size=min(len(candidates), max_fit_anchors), replace=False)
    scaler = StandardScaler().fit(matrix[chosen])
    transformed = scaler.transform(matrix[chosen])
    if encoder_type == "pca":
        encoder = PCA(n_components=dimensions, svd_solver="full").fit(transformed)
    else:
        # Mask 15% of standardized monthly-summary inputs and reconstruct the
        # unmasked state. Labels and OOT applications are not read here.
        masked = transformed.copy()
        masked[rng.random(masked.shape) < 0.15] = 0
        encoder = MLPRegressor(
            hidden_layer_sizes=(64, dimensions, 64), activation="relu",
            solver="adam", batch_size=min(512, len(chosen)), max_iter=80,
            early_stopping=len(chosen) >= 40, n_iter_no_change=8,
            random_state=17,
        ).fit(masked, transformed)
    provisional = {"scaler": scaler, "encoder": encoder, "encoder_type": encoder_type}
    embedded = _embedding(provisional, matrix[chosen])
    kmeans = KMeans(n_clusters=clusters, random_state=17, n_init=10).fit(embedded)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = {"scaler": scaler, "encoder": encoder, "encoder_type": encoder_type,
             "dimensions": dimensions, "kmeans": kmeans, "history_months": history_months,
             "pit_basis": apps["pit_basis"].iloc[0],
             "feature_names": names, "fit_max_anchor": apps["application_time"].max().isoformat(),
             "declared_earliest_application": earliest.isoformat()}
    model_path = out / "encoder.joblib"
    joblib.dump(model, model_path)
    version = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    manifest = {"encoder_version": version, "fit_max_anchor": model["fit_max_anchor"],
                "declared_earliest_application": model["declared_earliest_application"],
                "n_eligible_fit_anchors": int(usable.sum()), "n_fit_anchors": len(chosen),
                "encoder_type": encoder_type, "pit_basis": model["pit_basis"],
                "embedding_dimensions": dimensions,
                "clusters": clusters, "history_months": history_months,
                "input_features": names}
    if encoder_type == "autoencoder":
        manifest["training_iterations"] = int(encoder.n_iter_)
        manifest["training_loss"] = float(encoder.loss_)
        manifest["training_reached_iteration_limit"] = bool(encoder.n_iter_ >= encoder.max_iter)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def score(raw_glob: str, model_dir: str, output: str) -> dict:
    model_dir = Path(model_dir)
    model_path = model_dir / "encoder.joblib"
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(model_path.read_bytes()).hexdigest()[:16] != manifest["encoder_version"]:
        raise ValueError("Encoder artifact hash does not match manifest")
    model = joblib.load(model_path)  # Only load artifacts created in the trusted workspace.
    apps, matrix, names = summarize(raw_glob, model["history_months"])
    if apps["pit_basis"].nunique() != 1 or apps["pit_basis"].iloc[0] != model["pit_basis"]:
        raise ValueError("Scoring source adapter differs from fitted encoder")
    if names != model["feature_names"]:
        raise ValueError("Feature contract does not match fitted encoder")
    if apps["application_time"].min() <= pd.Timestamp(model["fit_max_anchor"]):
        raise ValueError("Scoring an application at or before the fit corpus cutoff")
    out = apps.copy()
    conventional = _conventional(apps, matrix, model["history_months"])
    for column in conventional.columns:
        out[column] = conventional[column]
    dim = model["dimensions"]
    clusters = model["kmeans"].n_clusters
    emb_cols = [f"scr_emb_{j:02d}" for j in range(dim)]
    dist_cols = [f"scr_cluster_distance_{j:02d}" for j in range(clusters)]
    extra_cols = ["scr_emb_norm", "scr_cluster_nearest_distance", "scr_cluster_margin"]
    for column in emb_cols + dist_cols + ["scr_cluster"] + extra_cols:
        out[column] = np.nan
    usable = out["coverage_status"].eq("USABLE").to_numpy()
    if usable.any():
        embedded = _embedding(model, matrix[usable])
        distances = model["kmeans"].transform(embedded)
        for j, column in enumerate(emb_cols):
            out.loc[usable, column] = embedded[:, j]
        for j, column in enumerate(dist_cols):
            out.loc[usable, column] = distances[:, j]
        out.loc[usable, "scr_cluster"] = model["kmeans"].predict(embedded)
        out.loc[usable, "scr_emb_norm"] = np.linalg.norm(embedded, axis=1)
        ordered = np.sort(distances, axis=1)
        out.loc[usable, "scr_cluster_nearest_distance"] = ordered[:, 0]
        out.loc[usable, "scr_cluster_margin"] = ordered[:, 1] - ordered[:, 0]
    out["encoder_version"] = manifest["encoder_version"]
    if out[["application_id", "product"]].duplicated().any():
        raise ValueError("Scored output has duplicate application keys")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    representation_columns = emb_cols + ["scr_cluster"] + dist_cols + extra_cols
    feature_columns = list(conventional.columns) + representation_columns
    feature_manifest = {"encoder_version": manifest["encoder_version"], "feature_columns": feature_columns,
                        "feature_groups": {"conventional": list(conventional.columns),
                                           "representation": representation_columns},
                        "n_applications": len(out), "status_counts": out["coverage_status"].value_counts().to_dict()}
    output_path.with_suffix(".manifest.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    return feature_manifest


def audit(features_glob: str) -> dict:
    con = duckdb.connect()
    con.from_parquet(features_glob).create_view("features")
    n, keys = con.execute("SELECT COUNT(*), COUNT(DISTINCT (application_id, product)) FROM features").fetchone()
    if n != keys:
        raise ValueError("Duplicate feature keys")
    result = con.execute("""
      SELECT product, DATE_TRUNC('month', application_time) application_month, split,
             customer_status, target_provenance, coverage_status, COUNT(*) n
      FROM features GROUP BY ALL ORDER BY application_month, product, coverage_status
    """).df()
    con.close()
    return {"applications": n, "coverage": result.to_dict(orient="records")}
