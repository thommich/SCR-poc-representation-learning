import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from scr_features.features import fit, score, summarize
from scr_features.sql import load_config, render_anchor_sql, render_fit_sql, render_sql


def configured(tmp_path):
    source = Path(__file__).parents[1] / "config" / "example.json"
    cfg = json.loads(source.read_text(encoding="utf-8"))
    for adapter in ("spine", "snapshot", "authorization"):
        for key, value in cfg[adapter].items():
            cfg[adapter][key] = "audit_db." + key if key == "table" else "col_" + key
    cfg["certification"] = {key: True for key in cfg["certification"]}
    cfg["application_months"] = [f"2025-{m:02d}" for m in range(1, 9)]
    cfg["source_partitions"] = ["20250101", "20250201"]
    cfg["source_origin"] = "M"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_sql_requires_certification_and_uses_strict_asof(tmp_path):
    source = Path(__file__).parents[1] / "config" / "example.json"
    source_config = json.loads(source.read_text(encoding="utf-8"))
    source_config["source_mode"] = "certified_ledger"
    blocked = tmp_path / "blocked.json"
    blocked.write_text(json.dumps(source_config), encoding="utf-8")
    with pytest.raises(ValueError, match="Certification gate"):
        load_config(blocked)
    cfg = load_config(configured(tmp_path))
    sql = render_sql(cfg, "2025-01", "s3://test-bucket/fresh/")
    assert "< p.application_time" in sql
    assert "source_rows" in sql
    assert "UNKNOWN_AVAILABILITY" in sql
    assert "AMBIGUOUS_VERSION" in sql
    assert "anomesdia IN ('20250101', '20250201')" in sql
    assert "cod_orig_risc_bace = 'M'" in sql
    assert "ASSUMED_CONSULTATION" in sql
    partitioned = render_sql(cfg, "2025-01", "s3://test-bucket/partitioned/",
                             partitioned_buckets=16)
    assert "partitioned_by = ARRAY['bucket']" in partitioned
    assert "balance_overdue," in partitioned
    fit_sql = render_fit_sql(cfg, "2024-12-31T23:59:59", "2025-01-01T00:00:00",
                             "s3://test-bucket/fit/", sample_mod=1000)
    assert "'PRETRAIN' product" in fit_sql
    assert "'UNLABELED' target_provenance" in fit_sql
    assert "col_application_id" not in fit_sql
    with pytest.raises(ValueError, match="strictly before"):
        render_fit_sql(cfg, "2025-01-01T00:00:00", "2025-01-01T00:00:00",
                       "s3://test-bucket/fit/")
    anchor_sql = render_anchor_sql(cfg, "2025-01", 1, "2024-12-01T00:00:00",
                                   "s3://test-bucket/anchor/")
    assert "parent_application_id" in anchor_sql
    assert "__anchor_1" in anchor_sql
    assert "DATE_ADD('month', -1" in anchor_sql
    certified_config = json.loads(Path(configured(tmp_path)).read_text(encoding="utf-8"))
    certified_config["source_mode"] = "certified_ledger"
    certified_path = tmp_path / "certified.json"
    certified_path.write_text(json.dumps(certified_config), encoding="utf-8")
    certified_sql = render_sql(load_config(certified_path), "2025-01", "s3://test-bucket/certified/")
    assert "CERTIFIED_LEDGER" in certified_sql
    assert "permitted_intervals" in certified_sql
    with pytest.raises(ValueError):
        render_sql(cfg, "2025-09", "s3://test-bucket/fresh/")


def make_extract(path, anchor, n=9, future=False, duplicate=False):
    rows = []
    anchor_ts = pd.Timestamp(anchor)
    for i in range(n):
        usable = i != n - 1
        for age in ((1, 2) if usable else (None,)):
            ref = (anchor_ts.to_period("M") - age).strftime("%Y%m") if age else None
            rows.append({
                "application_id": f"a{i}", "person_id": f"p{i}",
                "application_time": anchor_ts, "product": "CREDIARIO" if i % 2 else "CONSIG_EP",
                "split": "TRAIN", "expected_split": "TRAIN", "spine_duplicate_count": 1,
                "pit_basis": "ASSUMED_CONSULTATION",
                "authorization_status": "ASSUMED_FROM_SCR_EVENT" if usable else "UNKNOWN",
                "target_value": 0 if i % 2 else 1,
                "target_provenance": "OBSERVED", "customer_status": "CUSTOMER",
                "coverage_status": "USABLE" if usable else "UNKNOWN_AVAILABILITY",
                "report_months": 2 if usable else 0, "no_report_months": 0,
                "no_exposure_months": 0, "event_id": f"e{i}_{age}" if age else None,
                "scr_ref_month": ref,
                "available_at": anchor_ts + timedelta(days=1) if future and i == 0 else (
                    anchor_ts - timedelta(days=2) if age else pd.NaT),
                "snapshot_status": "REPORT" if age else None,
                "record_key": f"r{i}_{age}" if age else None,
                "modality": "202" if age else None, "institution": "96" if age else None,
                "origin": "M" if age else None,
                "balance_current": float(i + age) if age else None,
                "balance_overdue": float(i % 3) if age else None,
            })
    if duplicate:
        rows.append(rows[0].copy())
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_fit_score_keeps_population_and_freezes_cutoff(tmp_path):
    historical = make_extract(tmp_path / "historical.parquet", "2024-12-15")
    scoring = make_extract(tmp_path / "scoring.parquet", "2025-01-15")
    with pytest.raises(ValueError, match="strictly before"):
        fit(str(historical), "2024-12-15", str(tmp_path / "bad"), dimensions=2, clusters=2)
    manifest = fit(str(historical), "2025-01-01", str(tmp_path / "model"), dimensions=2, clusters=2)
    assert manifest["n_fit_anchors"] == 8
    result = score(str(scoring), str(tmp_path / "model"), str(tmp_path / "features.parquet"))
    assert result["n_applications"] == 9
    output = pd.read_parquet(tmp_path / "features.parquet")
    assert output[["application_id", "product"]].drop_duplicates().shape[0] == 9
    assert output.loc[output.coverage_status == "UNKNOWN_AVAILABILITY", "scr_emb_00"].isna().all()
    assert output.loc[output.coverage_status == "USABLE", "scr_emb_00"].notna().all()
    assert result["feature_groups"]["representation"]
    with pytest.raises(ValueError, match="at or before"):
        score(str(historical), str(tmp_path / "model"), str(tmp_path / "invalid.parquet"))
    changed = pd.read_parquet(scoring)
    changed["pit_basis"] = "CERTIFIED_LEDGER"
    changed_path = tmp_path / "different_adapter.parquet"
    changed.to_parquet(changed_path, index=False)
    with pytest.raises(ValueError, match="source adapter differs"):
        score(str(changed_path), str(tmp_path / "model"), str(tmp_path / "invalid.parquet"))
    fit(str(historical), "2025-01-01", str(tmp_path / "linear"),
        dimensions=2, clusters=2, encoder_type="pca")
    assert score(str(scoring), str(tmp_path / "linear"),
                 str(tmp_path / "linear.parquet"))["n_applications"] == 9


def test_rejects_future_and_duplicate_source_rows(tmp_path):
    future = make_extract(tmp_path / "future.parquet", "2025-01-15", future=True)
    with pytest.raises(ValueError, match="integrity violation"):
        summarize(str(future))
    dup = make_extract(tmp_path / "dup.parquet", "2025-01-15", duplicate=True)
    with pytest.raises(ValueError, match="Duplicated source"):
        summarize(str(dup))
