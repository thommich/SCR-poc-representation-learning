"""Rigid-transform-invariant geometry and monthly trajectory features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def _cols(frame: pd.DataFrame, prefix: str) -> list[str]:
    cols = sorted(c for c in frame if c.startswith(prefix) and c[len(prefix):].isdigit())
    if not cols:
        raise ValueError(f"No {prefix} coordinates")
    return cols


def _scale(points: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    take = points[rng.choice(len(points), min(len(points), 1024), replace=False)]
    if len(take) < 2:
        raise ValueError("Too few reference points")
    pairs = rng.integers(0, len(take), size=(min(4096, len(take) * 4), 2))
    distances = np.linalg.norm(take[pairs[:, 0]] - take[pairs[:, 1]], axis=1)
    positive = distances[distances > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def fit_geometry(fit_apps_path: str, fit_months_path: str, earliest_application: str,
                 out_dir: str, *, clusters: int = 8, seed: int = 17,
                 neighbor_sample: int = 512) -> dict:
    apps = pd.read_parquet(fit_apps_path)
    months = pd.read_parquet(fit_months_path)
    if pd.to_datetime(apps["application_time"]).max() >= pd.Timestamp(earliest_application):
        raise ValueError("Geometry fit corpus crosses earliest application")
    if apps["history_model_version"].nunique() != 1 or months["record_model_version"].nunique() != 1:
        raise ValueError("Mixed representation versions")
    if apps["record_model_version"].iloc[0] != months["record_model_version"].iloc[0]:
        raise ValueError("Month and application encoders disagree")
    hcols, zcols = _cols(apps, "h_"), _cols(months, "z_")
    h = apps[hcols].dropna().to_numpy(np.float32)
    z = months[zcols].dropna().to_numpy(np.float32)
    if min(len(h), len(z)) < clusters or clusters < 2:
        raise ValueError("Insufficient fit embeddings for centroids")
    rng = np.random.default_rng(seed)
    sample = h[rng.choice(len(h), min(len(h), neighbor_sample), replace=False)]
    h_km = KMeans(n_clusters=clusters, random_state=seed, n_init=10).fit(h)
    z_km = KMeans(n_clusters=clusters, random_state=seed, n_init=10).fit(z)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "references.npz"
    np.savez_compressed(path, h_centers=h_km.cluster_centers_, z_centers=z_km.cluster_centers_,
                        h_mean=h.mean(axis=0), z_mean=z.mean(axis=0),
                        h_scale=np.array(_scale(h, seed)), z_scale=np.array(_scale(z, seed)),
                        h_neighbors=sample)
    manifest = {"version": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                "history_model_version": apps["history_model_version"].iloc[0],
                "record_model_version": apps["record_model_version"].iloc[0],
                "pit_basis": apps["pit_basis"].iloc[0],
                "fit_max_anchor": pd.to_datetime(apps["application_time"]).max().isoformat(),
                "h_dim": len(hcols), "z_dim": len(zcols), "clusters": clusters,
                "h_fit_points": len(h), "z_fit_points": len(z), "seed": seed}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load(out_dir: str):
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    path = out / "references.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest()[:16] != manifest["version"]:
        raise ValueError("Geometry reference hash mismatch")
    return dict(np.load(path)), manifest


def _path_features(points: np.ndarray, months: np.ndarray, centers: np.ndarray, scale: float,
                   prefix: str) -> dict[str, float]:
    names = ["observations", "latest_speed", "mean_speed", "max_speed", "path_length",
             "net_displacement", "path_efficiency", "turn_cosine", "approach_speed",
             "region_transitions", "displacement_3m"]
    out = {f"{prefix}_{name}": np.nan for name in names}
    out[f"{prefix}_observations"] = float(len(points))
    if len(points) < 2:
        return out
    order = np.argsort(months)
    points, months = points[order], months[order]
    gaps = np.diff(months).astype(float)
    if (gaps <= 0).any():
        raise ValueError("Repeated or non-monotone trajectory anchors")
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1) / scale
    speeds = lengths / gaps
    path = lengths.sum()
    net = np.linalg.norm(points[-1] - points[0]) / scale
    current_region = np.linalg.norm(centers - points[-1], axis=1).argmin()
    center = centers[current_region]
    approach = (np.linalg.norm(points[-2] - center) - np.linalg.norm(points[-1] - center)) / scale / gaps[-1]
    assigned = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
    out.update({
        f"{prefix}_latest_speed": float(speeds[-1]),
        f"{prefix}_mean_speed": float(speeds.mean()),
        f"{prefix}_max_speed": float(speeds.max()),
        f"{prefix}_path_length": float(path),
        f"{prefix}_net_displacement": float(net),
        f"{prefix}_path_efficiency": float(net / path) if path > 0 else np.nan,
        f"{prefix}_approach_speed": float(approach),
        f"{prefix}_region_transitions": float(np.count_nonzero(np.diff(assigned))),
    })
    exact = np.flatnonzero(months == months[-1] - 3)
    if len(exact):
        out[f"{prefix}_displacement_3m"] = float(np.linalg.norm(points[-1] - points[exact[-1]]) / scale)
    if len(steps) > 1:
        a, b = steps[-2], steps[-1]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        out[f"{prefix}_turn_cosine"] = float(np.dot(a, b) / denom) if denom > 0 else np.nan
    return out


def score_geometry(app_path: str, month_path: str, geometry_dir: str, out_path: str) -> dict:
    ref, manifest = _load(geometry_dir)
    apps = pd.read_parquet(app_path)
    months = pd.read_parquet(month_path)
    if apps["history_model_version"].nunique() != 1 or apps["history_model_version"].iloc[0] != manifest["history_model_version"]:
        raise ValueError("Application encoder version mismatch")
    if len(months) and (months["record_model_version"].nunique() != 1 or
                        months["record_model_version"].iloc[0] != manifest["record_model_version"]):
        raise ValueError("Monthly encoder version mismatch")
    if apps["pit_basis"].nunique() != 1 or apps["pit_basis"].iloc[0] != manifest["pit_basis"]:
        raise ValueError("Source basis mismatch")
    if pd.to_datetime(apps["application_time"]).min() <= pd.Timestamp(manifest["fit_max_anchor"]):
        raise ValueError("Geometry scoring precedes fit cutoff")
    hcols, zcols = _cols(apps, "h_"), _cols(months, "z_")
    if len(hcols) != manifest["h_dim"] or len(zcols) != manifest["z_dim"]:
        raise ValueError("Embedding dimension mismatch")
    output = apps[["application_id", "product", "application_time", "pit_basis", "coverage_status"]].copy()
    output["geometry_version"] = manifest["version"]
    static_cols = [f"scr_h_sorted_distance_{j:02d}" for j in range(manifest["clusters"])] + [
        "scr_h_nearest_distance", "scr_h_centroid_margin", "scr_h_center_distance",
        "scr_h_neighbor_distance"]
    for column in static_cols:
        output[column] = np.nan
    valid = apps[hcols].notna().all(axis=1).to_numpy()
    if valid.any():
        h = apps.loc[valid, hcols].to_numpy(np.float32)
        distances = np.linalg.norm(h[:, None, :] - ref["h_centers"][None, :, :], axis=2) / float(ref["h_scale"])
        ordered = np.sort(distances, axis=1)
        for j in range(manifest["clusters"]):
            output.loc[valid, f"scr_h_sorted_distance_{j:02d}"] = ordered[:, j]
        output.loc[valid, "scr_h_nearest_distance"] = ordered[:, 0]
        output.loc[valid, "scr_h_centroid_margin"] = ordered[:, 1] - ordered[:, 0]
        output.loc[valid, "scr_h_center_distance"] = np.linalg.norm(h - ref["h_mean"], axis=1) / float(ref["h_scale"])
        neighbor = np.linalg.norm(h[:, None, :] - ref["h_neighbors"][None, :, :], axis=2)
        output.loc[valid, "scr_h_neighbor_distance"] = neighbor.min(axis=1) / float(ref["h_scale"])
    monthly_groups = {key: group for key, group in months.groupby(["application_id", "product"], sort=False)}
    trajectory = []
    for key in zip(apps["application_id"], apps["product"]):
        group = monthly_groups.get(key)
        if group is None:
            values = _path_features(np.empty((0, len(zcols))), np.array([]),
                                    ref["z_centers"], float(ref["z_scale"]), "scr_z")
        else:
            # Chronological index of the SCR reference month, independent of consultation date.
            period = pd.PeriodIndex(group["scr_ref_month"].astype(str), freq="M")
            month_number = period.year * 12 + period.month
            values = _path_features(group[zcols].to_numpy(np.float32), np.asarray(month_number),
                                    ref["z_centers"], float(ref["z_scale"]), "scr_z")
        trajectory.append(values)
    trajectory_frame = pd.DataFrame(trajectory, index=output.index)
    output = pd.concat([output, trajectory_frame], axis=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(out_path, index=False)
    blocks = {"static_geometry": static_cols, "reference_trajectory": list(trajectory_frame.columns)}
    Path(out_path).with_suffix(".manifest.json").write_text(json.dumps({"geometry_version": manifest["version"],
        "feature_groups": blocks, "applications": len(output)}, indent=2), encoding="utf-8")
    return {"applications": len(output), "geometry_version": manifest["version"], "feature_groups": blocks}


def score_rolling_trajectory(anchor_apps_path: str | None, main_apps_path: str, geometry_dir: str,
                             out_path: str) -> dict:
    ref, manifest = _load(geometry_dir)
    main = pd.read_parquet(main_apps_path)
    if anchor_apps_path is None:
        anchors = pd.DataFrame()
        hcols = [f"h_{j:02d}" for j in range(manifest["h_dim"])]
        groups = {}
    else:
        anchors = pd.read_parquet(anchor_apps_path)
        required = {"parent_application_id", "anchor_time", "application_id", "product", "history_model_version"}
        if required - set(anchors):
            raise ValueError("Rolling anchors require parent_application_id and anchor_time")
        if anchors["history_model_version"].nunique() != 1 or anchors["history_model_version"].iloc[0] != manifest["history_model_version"]:
            raise ValueError("Rolling encoder version mismatch")
        hcols = _cols(anchors, "h_")
        if len(hcols) != manifest["h_dim"]:
            raise ValueError("Rolling embedding dimension mismatch")
        anchor_times = pd.to_datetime(anchors["anchor_time"])
        if (anchor_times <= pd.Timestamp(manifest["fit_max_anchor"])).any():
            raise ValueError("Rolling anchor precedes geometry fit cutoff")
        if anchors["pit_basis"].nunique() != 1 or anchors["pit_basis"].iloc[0] != manifest["pit_basis"]:
            raise ValueError("Rolling source basis mismatch")
        main_time = main.set_index(["application_id", "product"])["application_time"]
        parent_keys = pd.MultiIndex.from_arrays([anchors["parent_application_id"], anchors["product"]])
        parent_times = pd.to_datetime(parent_keys.map(main_time))
        if parent_times.isna().any() or (anchor_times >= parent_times).any():
            raise ValueError("Rolling anchor is missing its parent or is not earlier than application")
        groups = {key: group for key, group in anchors.groupby(["parent_application_id", "product"], sort=False)}
    features = []
    for app in main.itertuples(index=False):
        group = groups.get((app.application_id, app.product))
        if group is None:
            values = _path_features(np.empty((0, len(hcols))), np.array([]),
                                    ref["h_centers"], float(ref["h_scale"]), "scr_rolling_h")
        else:
            group = group.dropna(subset=hcols)
            timestamps = pd.to_datetime(group["anchor_time"])
            month_number = timestamps.dt.year.to_numpy() * 12 + timestamps.dt.month.to_numpy()
            values = _path_features(group[hcols].to_numpy(np.float32), month_number,
                                    ref["h_centers"], float(ref["h_scale"]), "scr_rolling_h")
        features.append(values)
    output = pd.concat([main[["application_id", "product"]].reset_index(drop=True),
                        pd.DataFrame(features)], axis=1)
    output["geometry_version"] = manifest["version"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(out_path, index=False)
    return {"applications": len(output), "feature_columns": list(output.columns[2:-1])}
