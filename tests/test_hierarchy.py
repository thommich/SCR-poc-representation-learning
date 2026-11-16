from pathlib import Path
from datetime import timedelta

import pandas as pd
import pytest

from scr_features.hierarchy import (encode_applications, encode_months,
                                    fit_history, fit_record_encoder)
from scr_features.geometry import fit_geometry, score_geometry


def extract(path: Path, anchor: str, n: int = 12):
    rows = []
    timestamp = pd.Timestamp(anchor)
    for i in range(n):
        for age in (1, 2, 3):
            rows.append({
                "application_id": f"app_{i}", "person_id": f"person_{i}",
                "application_time": timestamp, "product": "CREDIARIO", "split": "FIT" if timestamp.year == 2024 else "TRAIN",
                "expected_split": "FIT" if timestamp.year == 2024 else "TRAIN",
                "spine_duplicate_count": 1, "pit_basis": "ASSUMED_CONSULTATION",
                "authorization_status": "ASSUMED_FROM_SCR_EVENT", "target_value": 1 if i % 3 == 0 else 0,
                "target_provenance": "UNLABELED" if timestamp.year == 2024 else "OBSERVED",
                "customer_status": "CUSTOMER", "coverage_status": "USABLE",
                "report_months": 3, "no_report_months": 0, "no_exposure_months": 0,
                "event_id": f"event_{i}_{age}",
                "scr_ref_month": (timestamp.to_period("M") - age).strftime("%Y%m"),
                "available_at": timestamp - timedelta(days=1),
                "snapshot_status": "REPORT", "record_key": f"record_{i}_{age}",
                "modality": "202" if i % 2 else "203", "institution": "96",
                "origin": "M", "balance_current": 100.0 + i + age,
                "balance_overdue": float((i + age) % 5),
            })
    pd.DataFrame(rows).to_parquet(path, index=False)
    return str(path)


def test_month_record_and_temporal_variants(tmp_path):
    fit_raw = extract(tmp_path / "fit.parquet", "2024-12-15")
    score_raw = extract(tmp_path / "score.parquet", "2025-01-15")
    record_dir = str(tmp_path / "record")
    fit_record_encoder(fit_raw, "2025-01-01", record_dir, epochs=2, batch_months=16)
    fit_months = str(tmp_path / "fit_months.parquet")
    fit_apps = str(tmp_path / "fit_apps.parquet")
    score_months = str(tmp_path / "score_months.parquet")
    score_apps = str(tmp_path / "score_apps.parquet")
    encode_months(fit_raw, record_dir, fit_months, fit_apps, for_fit=True)
    encode_months(score_raw, record_dir, score_months, score_apps)
    assert pd.read_parquet(fit_months).shape[0] == 36
    for variant in ("T0", "T1", "T2", "T3", "T4"):
        model_dir = str(tmp_path / variant)
        fit_history(fit_months, fit_apps, record_dir, "2025-01-01", model_dir,
                    variant=variant, epochs=2, batch_size=6)
        output = str(tmp_path / f"{variant}_app.parquet")
        result = encode_applications(score_months, score_apps, model_dir, output)
        assert result["embeddable"] == 12
        assert pd.read_parquet(output)["h_00"].notna().all()
        if variant == "T1":
            fit_output = str(tmp_path / "fit_application.parquet")
            encode_applications(fit_months, fit_apps, model_dir, fit_output, for_fit=True)
            geom_dir = str(tmp_path / "geometry")
            fit_geometry(fit_output, fit_months, "2025-01-01", geom_dir, clusters=2)
            geom_out = str(tmp_path / "geometry.parquet")
            score_geometry(output, score_months, geom_dir, geom_out)
            geometry = pd.read_parquet(geom_out)
            assert len(geometry) == 12
            assert geometry["scr_h_nearest_distance"].notna().all()
            assert geometry["scr_z_latest_speed"].notna().all()


def test_record_encoder_rejects_origin_drift(tmp_path):
    fit_raw = extract(tmp_path / "fit.parquet", "2024-12-15")
    score_raw = extract(tmp_path / "score.parquet", "2025-01-15")
    record_dir = str(tmp_path / "record")
    manifest = fit_record_encoder(fit_raw, "2025-01-01", record_dir, epochs=1)
    assert manifest["source_origin"] == "M"
    changed = pd.read_parquet(score_raw)
    changed["origin"] = "I"
    changed.to_parquet(score_raw, index=False)
    with pytest.raises(ValueError, match="SCR origin differs"):
        encode_months(score_raw, record_dir, str(tmp_path / "months.parquet"),
                      str(tmp_path / "apps.parquet"))
