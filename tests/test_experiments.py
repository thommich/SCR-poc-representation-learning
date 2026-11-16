import json

import numpy as np
import pandas as pd

from scr_features.demo import make_demo_data
from scr_features.evaluation import run_ladder, temporal_probe
from scr_features.features import export_conventional
from scr_features.geometry import (_path_features, fit_geometry, score_geometry,
                                   score_rolling_trajectory)
from scr_features.hierarchy import (encode_applications, encode_months,
                                    fit_history, fit_record_encoder)
from scr_features.stability import compare_embeddings, compare_feature_ranks


def test_full_synthetic_experiment(tmp_path):
    paths = make_demo_data(str(tmp_path / "data"), applicants_per_product_month=8)
    record_dir = str(tmp_path / "record")
    fit_record_encoder(paths["fit_raw"], paths["earliest_application"], record_dir,
                       epochs=1, batch_months=128)
    fit_months, fit_apps = str(tmp_path / "fit_m.parquet"), str(tmp_path / "fit_a.parquet")
    score_months, score_apps = str(tmp_path / "score_m.parquet"), str(tmp_path / "score_a.parquet")
    anchor_months, anchor_apps = str(tmp_path / "anchor_m.parquet"), str(tmp_path / "anchor_a.parquet")
    encode_months(paths["fit_raw"], record_dir, fit_months, fit_apps, for_fit=True)
    encode_months(paths["application_raw"], record_dir, score_months, score_apps)
    encode_months(paths["anchor_raw"], record_dir, anchor_months, anchor_apps)
    history_dir = str(tmp_path / "T1")
    fit_history(fit_months, fit_apps, record_dir, paths["earliest_application"],
                history_dir, variant="T1", epochs=1, batch_size=64)
    fit_emb, score_emb, anchor_emb = [str(tmp_path / f"{name}.parquet")
                                      for name in ("fit_emb", "score_emb", "anchor_emb")]
    encode_applications(fit_months, fit_apps, history_dir, fit_emb, for_fit=True)
    encode_applications(score_months, score_apps, history_dir, score_emb)
    encode_applications(anchor_months, anchor_apps, history_dir, anchor_emb)
    assert {"parent_application_id", "anchor_time"} <= set(pd.read_parquet(anchor_emb))
    probe = temporal_probe(paths["incumbent"], {"T1": score_emb},
                           ["incumbent_score_a", "incumbent_score_b"],
                           product="CREDIARIO", validation_month="2025-05",
                           train_provenances=["OBSERVED"],
                           model_params={"n_estimators": 20, "min_child_samples": 2})
    assert "T1" in probe["variants"]
    geom_dir = str(tmp_path / "geom")
    fit_geometry(fit_emb, fit_months, paths["earliest_application"], geom_dir, clusters=3)
    geom = str(tmp_path / "geom.parquet")
    score_geometry(score_emb, score_months, geom_dir, geom)
    rolling = str(tmp_path / "rolling.parquet")
    score_rolling_trajectory(anchor_emb, score_emb, geom_dir, rolling)
    conventional = str(tmp_path / "conventional.parquet")
    export_conventional(paths["application_raw"], conventional)
    report = run_ladder(paths["incumbent"], conventional, score_emb, geom,
                        str(tmp_path / "geom.manifest.json"), str(tmp_path / "models"),
                        ["incumbent_score_a", "incumbent_score_b"], product="CREDIARIO",
                        train_provenances=["OBSERVED"], validation_month="2025-05",
                        rolling_path=rolling, n_bootstrap=5,
                        model_params={"n_estimators": 20, "min_child_samples": 2})
    assert set(report["models"]) == {"B0", "B1", "B2", "B3", "B4", "B5", "B6"}
    assert report["population"] == 64
    assert json.loads((tmp_path / "models" / "scorecard.json").read_text())["oot_observed"] == 24


def test_path_geometry_survives_rigid_transform():
    points = np.array([[0., 1.], [1., 1.], [2., 2.], [3., 2.]])
    centers = np.array([[1., 0.], [3., 3.]])
    dates = np.array([10, 11, 13, 14])
    original = _path_features(points, dates, centers, 2.0, "path")
    rotation = np.array([[0., -1.], [1., 0.]])
    translation = np.array([8., -5.])
    moved = _path_features(points @ rotation + translation, dates,
                           centers @ rotation + translation, 2.0, "path")
    for key in original:
        assert np.isclose(original[key], moved[key], equal_nan=True), key


def test_cross_retrain_geometry_metrics_for_rigid_rotation():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(40, 2))
    rotation = np.array([[0., -1.], [1., 0.]])
    left = pd.DataFrame({"application_id": [str(i) for i in range(len(points))],
                         "h_00": points[:, 0], "h_01": points[:, 1],
                         "distance": np.linalg.norm(points, axis=1)})
    transformed = points @ rotation + np.array([7., -4.])
    right = pd.DataFrame({"application_id": left["application_id"],
                          "h_00": transformed[:, 0], "h_01": transformed[:, 1],
                          "distance": left["distance"]})
    result = compare_embeddings(left, right, keys=["application_id"], prefix="h_", neighbors=3)
    assert result["procrustes_residual"] < 1e-10
    assert result["pairwise_distance_spearman"] > 0.999
    assert result["neighbor_overlap"] > 0.999
    feature = compare_feature_ranks(left, right, keys=["application_id"],
                                    feature_columns=["distance"])
    assert feature["distance"]["spearman"] > 0.999
