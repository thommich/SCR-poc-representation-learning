"""Cross-retrain geometry checks on a shared, pre-cutoff anchor panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


def compare_embeddings(left: pd.DataFrame, right: pd.DataFrame, *,
                       keys: list[str], prefix: str, max_points: int = 5000,
                       neighbors: int = 5, seed: int = 17) -> dict:
    cols = sorted(c for c in left if c.startswith(prefix) and c[len(prefix):].isdigit())
    if not cols or not set(cols) <= set(right):
        raise ValueError("Embedding coordinate sets differ")
    a = left[keys + cols].dropna().merge(right[keys + cols].dropna(), on=keys,
                                          validate="one_to_one", suffixes=("_a", "_b"))
    if len(a) < max(neighbors + 2, 4):
        raise ValueError("Too few shared anchor embeddings")
    rng = np.random.default_rng(seed)
    if len(a) > max_points:
        a = a.iloc[rng.choice(len(a), max_points, replace=False)].reset_index(drop=True)
    x = a[[f"{c}_a" for c in cols]].to_numpy(float)
    y = a[[f"{c}_b" for c in cols]].to_numpy(float)
    xc, yc = x - x.mean(0), y - y.mean(0)
    rotation, _ = orthogonal_procrustes(xc, yc)
    procrustes = float(np.linalg.norm(xc @ rotation - yc) / max(np.linalg.norm(yc), 1e-12))
    pairs = rng.integers(0, len(a), size=(min(20000, len(a) * 8), 2))
    dx = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
    dy = np.linalg.norm(y[pairs[:, 0]] - y[pairs[:, 1]], axis=1)
    distance_rank = float(spearmanr(dx, dy).statistic)
    k = min(neighbors, len(a) - 1)
    nx = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x, return_distance=False)[:, 1:]
    ny = NearestNeighbors(n_neighbors=k + 1).fit(y).kneighbors(y, return_distance=False)[:, 1:]
    overlap = float(np.mean([len(set(row_x) & set(row_y)) / k for row_x, row_y in zip(nx, ny)]))
    return {"anchors": len(a), "procrustes_residual": procrustes,
            "pairwise_distance_spearman": distance_rank, "neighbor_overlap": overlap}


def compare_feature_ranks(left: pd.DataFrame, right: pd.DataFrame, *,
                          keys: list[str], feature_columns: list[str]) -> dict:
    if not feature_columns:
        raise ValueError("Select matched geometry features")
    matched = left[keys + feature_columns].merge(right[keys + feature_columns], on=keys,
                                                  validate="one_to_one", suffixes=("_a", "_b"))
    result = {}
    for feature in feature_columns:
        a, b = matched[f"{feature}_a"], matched[f"{feature}_b"]
        valid = a.notna() & b.notna()
        if valid.sum() < 4:
            result[feature] = {"n": int(valid.sum()), "spearman": None, "decile_agreement": None}
            continue
        ranks_a = a[valid].rank(pct=True).to_numpy()
        ranks_b = b[valid].rank(pct=True).to_numpy()
        result[feature] = {
            "n": int(valid.sum()), "spearman": float(spearmanr(a[valid], b[valid]).statistic),
            "decile_agreement": float(np.mean(np.minimum(9, (ranks_a * 10).astype(int)) ==
                                              np.minimum(9, (ranks_b * 10).astype(int)))),
        }
    return result
