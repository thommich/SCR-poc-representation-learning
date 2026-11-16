"""Small synthetic-only fixtures for notebook execution tests."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _rows(application_id, person_id, when, product, split, target, *,
          parent=None, anchor_time=None, seed=0):
    result = []
    for age in (1, 2, 3, 4):
        for record in (0, 1):
            amount = 100 + (seed % 17) * 11 + age * 4 + record * 7
            overdue = float((seed + age + record) % 8)
            result.append({
                "application_id": application_id, "parent_application_id": parent,
                "person_id": person_id, "application_time": when,
                "anchor_time": anchor_time, "product": product, "split": split,
                "expected_split": split, "spine_duplicate_count": 1,
                "pit_basis": "ASSUMED_CONSULTATION",
                "authorization_status": "ASSUMED_FROM_SCR_EVENT",
                "target_value": target,
                "target_provenance": "UNLABELED" if target is None else "OBSERVED",
                "customer_status": "CUSTOMER" if seed % 2 else "NONCUSTOMER",
                "coverage_status": "USABLE", "report_months": 4,
                "no_report_months": 0, "no_exposure_months": 0,
                "event_id": f"{application_id}_event_{age}",
                "scr_ref_month": (when.to_period("M") - age).strftime("%Y%m"),
                "available_at": when - timedelta(days=1),
                "replaced_at": pd.NaT, "snapshot_status": "REPORT",
                "source_partition": "20250101", "acquisition": "Consulta Lote",
                "protocol": "123", "consultation_raw": str(when - timedelta(days=1)),
                "record_key": f"{application_id}_record_{age}_{record}",
                "modality": "202" if record else "203",
                "institution": "96" if seed % 2 else "42",
                "origin": "M", "balance_current": float(amount),
                "balance_overdue": overdue,
            })
    return result


def make_demo_data(root: str, *, applicants_per_product_month: int = 16) -> dict:
    """Create labelled synthetic data; never interpret its metrics as POC results."""
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    fit = []
    fit_time = pd.Timestamp("2024-12-15 12:00:00")
    for i in range(64):
        fit.extend(_rows(f"FIT_{i}", f"FIT_PERSON_{i}", fit_time, "PRETRAIN", "FIT", None, seed=i))
    fit_path = out / "fit.parquet"
    pd.DataFrame(fit).to_parquet(fit_path, index=False)
    score = []
    anchors = []
    incumbent = []
    rng = np.random.default_rng(17)
    for month in range(1, 9):
        when = pd.Timestamp(f"2025-{month:02d}-15 12:00:00")
        split = "TRAIN" if month <= 5 else "OOT"
        for product in ("CREDIARIO", "CONSIG_EP"):
            for i in range(applicants_per_product_month):
                app_id = f"{product}_{month}_{i}"
                person = f"PERSON_{product}_{i}"
                risk = 0.2 * (i % 4) + 0.05 * month + 0.15 * (product == "CONSIG_EP")
                target = int(rng.random() < 0.15 + risk / 2)
                score.extend(_rows(app_id, person, when, product, split, target,
                                   seed=i + month * 11))
                incumbent.append({"application_id": app_id, "product": product,
                                  "person_id": person, "application_time": when,
                                  "split": split, "target_value": target,
                                  "target_provenance": "OBSERVED",
                                  "customer_status": "CUSTOMER" if i % 2 else "NONCUSTOMER",
                                  "incumbent_score_a": float(risk + rng.normal(0, 0.2)),
                                  "incumbent_score_b": float(i % 5)})
                for offset in (1, 2, 3):
                    anchor = when - pd.DateOffset(months=offset)
                    if anchor <= fit_time:
                        continue
                    anchors.extend(_rows(f"{app_id}__anchor_{offset}", person, anchor,
                                         product, split, None, parent=app_id,
                                         anchor_time=anchor, seed=i + month * 11 - offset))
    score_path = out / "applications.parquet"
    anchor_path = out / "anchors.parquet"
    incumbent_path = out / "incumbent.parquet"
    pd.DataFrame(score).to_parquet(score_path, index=False)
    pd.DataFrame(anchors).to_parquet(anchor_path, index=False)
    pd.DataFrame(incumbent).to_parquet(incumbent_path, index=False)
    return {"fit_raw": str(fit_path), "application_raw": str(score_path),
            "anchor_raw": str(anchor_path), "incumbent": str(incumbent_path),
            "earliest_application": "2025-01-01T00:00:00", "validation_month": "2025-05"}
