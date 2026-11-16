"""Frozen record -> month -> application representations for the SCR experiments.

All fit routines accept only pre-cutoff, unlabeled extracts. The same monthly
vectors feed every temporal history aggregator. This module deliberately uses
no application-ID embedding table and no credit labels.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from .features import META, _assert_raw, _read

torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))


def _save(model: nn.Module, payload: dict, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **payload}, out / "model.pt")
    version = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()[:16]
    manifest = {k: v for k, v in payload.items() if k != "state_dict"}
    manifest["version"] = version
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load(out_dir: str, cls):
    root = Path(out_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256((root / "model.pt").read_bytes()).hexdigest()[:16] != manifest["version"]:
        raise ValueError("Artifact hash mismatch")
    data = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    model = cls(data)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, manifest


def _raw_frames(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = _read(path)
    _assert_raw(con)
    available_columns = {row[1] for row in con.execute("PRAGMA table_info('raw')").fetchall()}
    optional = [c for c in ("parent_application_id", "anchor_time") if c in available_columns]
    apps = con.execute(f"SELECT DISTINCT {', '.join(META + optional)} FROM raw").df()
    records = con.execute("""
      SELECT application_id, product, scr_ref_month, event_id, record_key, modality,
             institution, origin, balance_current, balance_overdue, available_at
      FROM raw WHERE snapshot_status = 'REPORT' AND record_key IS NOT NULL
    """).df()
    con.close()
    apps["application_time"] = pd.to_datetime(apps["application_time"])
    if not records.empty:
        records["available_at"] = pd.to_datetime(records["available_at"])
        if (records.groupby(["application_id", "product", "scr_ref_month"])["event_id"]
                .nunique().gt(1).any()):
            raise ValueError("Multiple selected SCR events for one application/reference month")
    return apps, records


def _vocab(values: pd.Series, cap: int = 4096) -> dict[str, int]:
    counts = values.fillna("<MISSING>").astype(str).value_counts()
    return {token: i + 1 for i, token in enumerate(counts.index[:cap])}


def _signed_log(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0)
    return np.sign(values) * np.log1p(np.abs(values))


class MonthRecords:
    def __init__(self, records: pd.DataFrame, vocabs: dict):
        if records.empty:
            raise ValueError("No report records in extract")
        frame = records.sort_values(["application_id", "product", "scr_ref_month", "record_key"],
                                    kind="stable").reset_index(drop=True)
        groups = frame.groupby(["application_id", "product", "scr_ref_month"], sort=False, dropna=False)
        sizes = groups.size().to_numpy()
        self.starts = np.r_[0, sizes.cumsum()]
        self.months = groups.agg(
            available_at=("available_at", "max"),
            balance_current=("balance_current", "sum"),
            balance_overdue=("balance_overdue", "sum"),
            record_count=("record_key", "size"),
            modality_count=("modality", "nunique"),
        ).reset_index()
        self.numeric = np.column_stack([
            _signed_log(frame["balance_current"].to_numpy()),
            _signed_log(frame["balance_overdue"].to_numpy()),
        ]).astype(np.float32)
        self.modality = frame["modality"].fillna("<MISSING>").astype(str).map(vocabs["modality"]).fillna(0).to_numpy(np.int64)
        self.institution = frame["institution"].fillna("<MISSING>").astype(str).map(vocabs["institution"]).fillna(0).to_numpy(np.int64)
        self.target = np.column_stack([
            _signed_log(self.months["balance_current"].to_numpy()),
            _signed_log(self.months["balance_overdue"].to_numpy()),
            np.log1p(self.months["record_count"].to_numpy(np.float32)),
            np.log1p(self.months["modality_count"].to_numpy(np.float32)),
        ]).astype(np.float32)

    def batch(self, indices: np.ndarray):
        rec_indices = np.concatenate([np.arange(self.starts[i], self.starts[i + 1]) for i in indices])
        local = np.repeat(np.arange(len(indices)),
                          [self.starts[i + 1] - self.starts[i] for i in indices])
        return (
            torch.from_numpy(self.numeric[rec_indices]),
            torch.from_numpy(self.modality[rec_indices]),
            torch.from_numpy(self.institution[rec_indices]),
            torch.from_numpy(local.astype(np.int64)),
            torch.from_numpy(self.target[indices]),
        )


class RecordMonthEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim = cfg["month_dim"]
        self.modality = nn.Embedding(cfg["modality_size"], 8)
        self.institution = nn.Embedding(cfg["institution_size"], 8)
        self.record = nn.Sequential(nn.Linear(18, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.month = nn.Sequential(nn.Linear(33, 64), nn.ReLU(), nn.Linear(64, dim))
        self.decoder = nn.Linear(dim, 4)

    def forward(self, numeric, modality, institution, month_index, n_months, mask_rate=0.0):
        x = torch.cat([numeric, self.modality(modality), self.institution(institution)], dim=1)
        encoded = self.record(x)
        if mask_rate:
            keep = (torch.rand(len(encoded), 1, device=encoded.device) >= mask_rate).float()
            encoded = encoded * keep
        else:
            keep = torch.ones(len(encoded), 1, device=encoded.device)
        pooled = torch.zeros(n_months, 32, device=encoded.device).index_add_(0, month_index, encoded)
        counts = torch.zeros(n_months, 1, device=encoded.device).index_add_(0, month_index, keep)
        pooled = pooled / counts.clamp_min(1)
        z = self.month(torch.cat([pooled, torch.log1p(counts)], dim=1))
        return z, self.decoder(z)


def fit_record_encoder(raw_path: str, earliest_application: str, out_dir: str, *,
                       month_dim: int = 16, epochs: int = 25, batch_months: int = 256,
                       seed: int = 17) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    apps, records = _raw_frames(raw_path)
    if apps.empty or apps["application_time"].max() >= pd.Timestamp(earliest_application):
        raise ValueError("Record encoder fit data must precede the earliest application")
    if apps["pit_basis"].nunique() != 1:
        raise ValueError("Mixed source bases")
    if records["origin"].isna().any() or records["origin"].nunique() != 1:
        raise ValueError("Record encoder requires one verified SCR origin code")
    vocabs = {"modality": _vocab(records["modality"]), "institution": _vocab(records["institution"])}
    data = MonthRecords(records, vocabs)
    cfg = {"month_dim": month_dim, "modality_size": len(vocabs["modality"]) + 1,
           "institution_size": len(vocabs["institution"]) + 1}
    model = RecordMonthEncoder(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    rng = np.random.default_rng(seed)
    losses = []
    started = time.perf_counter()
    for _ in range(epochs):
        order = rng.permutation(len(data.months))
        model.train()
        batch_losses = []
        for chunk in np.array_split(order, max(1, int(np.ceil(len(order) / batch_months)))):
            numeric, mod, inst, month_idx, target = data.batch(chunk)
            _, prediction = model(numeric, mod, inst, month_idx, len(chunk), mask_rate=0.15)
            loss = nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        losses.append(float(np.mean(batch_losses)))
    payload = {"kind": "record_month", "model_config": cfg, "vocabs": vocabs,
               "fit_max_anchor": apps["application_time"].max().isoformat(),
               "pit_basis": apps["pit_basis"].iloc[0], "source_origin": records["origin"].iloc[0],
               "seed": seed, "epochs": epochs,
               "monthly_fit_rows": len(data.months), "final_reconstruction_loss": losses[-1],
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "training_seconds": time.perf_counter() - started}
    return _save(model, payload, out_dir)


def encode_months(raw_path: str, record_model_dir: str, out_months: str,
                  out_apps: str, *, batch_months: int = 512,
                  for_fit: bool = False) -> dict:
    started = time.perf_counter()
    model, manifest = _load(record_model_dir, lambda d: RecordMonthEncoder(d["model_config"]))
    apps, records = _raw_frames(raw_path)
    if apps["pit_basis"].nunique() != 1 or apps["pit_basis"].iloc[0] != manifest["pit_basis"]:
        raise ValueError("Source basis differs from fitted record encoder")
    if len(records) and (records["origin"].isna().any() or
                         records["origin"].nunique() != 1 or
                         records["origin"].iloc[0] != manifest["source_origin"]):
        raise ValueError("SCR origin differs from fitted record encoder")
    if for_fit:
        if apps["application_time"].max() > pd.Timestamp(manifest["fit_max_anchor"]):
            raise ValueError("Fit materialization includes later applications")
    elif apps["application_time"].min() <= pd.Timestamp(manifest["fit_max_anchor"]):
        raise ValueError("Monthly scoring precedes record encoder fit cutoff")
    Path(out_apps).parent.mkdir(parents=True, exist_ok=True)
    apps.to_parquet(out_apps, index=False)
    if records.empty:
        month_frame = pd.DataFrame(columns=["application_id", "product", "scr_ref_month",
                                            "available_at", "age_months", "record_model_version"] +
                                   [f"z_{j:02d}" for j in range(manifest["model_config"]["month_dim"])])
    else:
        data = MonthRecords(records, manifest["vocabs"])
        vectors = np.empty((len(data.months), manifest["model_config"]["month_dim"]), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(data.months), batch_months):
                indices = np.arange(start, min(start + batch_months, len(data.months)))
                numeric, mod, inst, month_idx, _ = data.batch(indices)
                z, _ = model(numeric, mod, inst, month_idx, len(indices))
                vectors[indices] = z.numpy()
        month_frame = data.months.copy()
        app_times = apps.set_index(["application_id", "product"])["application_time"]
        month_frame["application_time"] = pd.MultiIndex.from_frame(month_frame[["application_id", "product"]]).map(app_times)
        ref = pd.PeriodIndex(month_frame["scr_ref_month"].astype(str), freq="M")
        app = pd.to_datetime(month_frame["application_time"]).dt.to_period("M")
        month_frame["age_months"] = (app.dt.year.to_numpy() - ref.year) * 12 + (app.dt.month.to_numpy() - ref.month)
        for j in range(vectors.shape[1]):
            month_frame[f"z_{j:02d}"] = vectors[:, j]
        month_frame["record_model_version"] = manifest["version"]
        month_frame = month_frame.drop(columns="application_time")
    Path(out_months).parent.mkdir(parents=True, exist_ok=True)
    month_frame.to_parquet(out_months, index=False)
    return {"applications": len(apps), "represented_months": len(month_frame),
            "record_model_version": manifest["version"],
            "encoding_seconds": time.perf_counter() - started}


def sequence_arrays(months_path: str, apps_path: str, history_months: int = 12):
    apps = pd.read_parquet(apps_path)
    months = pd.read_parquet(months_path)
    if apps[["application_id", "product"]].duplicated().any():
        raise ValueError("Duplicate application keys")
    cols = sorted((c for c in months.columns if c.startswith("z_") and c[2:].isdigit()))
    if not cols:
        raise ValueError("No monthly embedding dimensions")
    if len(months) and months["record_model_version"].nunique() != 1:
        raise ValueError("Mixed monthly encoder versions")
    x = np.zeros((len(apps), history_months, len(cols)), np.float32)
    mask = np.zeros((len(apps), history_months), bool)
    age = np.zeros((len(apps), history_months), np.float32)
    index = pd.MultiIndex.from_frame(apps[["application_id", "product"]])
    positions = index.get_indexer(pd.MultiIndex.from_frame(months[["application_id", "product"]]))
    if (positions < 0).any():
        raise ValueError("Monthly embedding lacks application")
    for j, row in enumerate(months.itertuples(index=False)):
        lag = int(row.age_months)
        if not 1 <= lag <= history_months:
            raise ValueError("Monthly embedding outside history window")
        slot = history_months - lag
        if mask[positions[j], slot]:
            raise ValueError("Duplicate application/reference month")
        x[positions[j], slot] = months.iloc[j][cols].to_numpy(np.float32)
        mask[positions[j], slot] = True
        age[positions[j], slot] = lag
    return apps, x, mask, age, cols


class HistoryEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.variant = cfg["variant"]
        self.month_dim = cfg["month_dim"]
        self.out_dim = cfg["out_dim"]
        if self.variant == "T0":
            self.pool = nn.Sequential(nn.Linear(self.month_dim, 64), nn.ReLU(), nn.Linear(64, self.out_dim))
        elif self.variant == "T1":
            self.token = nn.Sequential(nn.Linear(self.month_dim + 1, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
            self.pool = nn.Linear(32, self.out_dim)
        else:
            input_dim = self.month_dim + (2 if self.variant == "T4" else 1)
            self.gru = nn.GRU(input_dim, 32, batch_first=True)
            self.project = nn.Linear(32, self.out_dim)
        self.head = nn.Linear(self.out_dim, self.month_dim)

    def forward(self, x, mask, age, *, shuffle=False, generator=None):
        if self.variant == "T0":
            pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
            h = self.pool(pooled)
        elif self.variant == "T1":
            token = self.token(torch.cat([x, age.unsqueeze(-1) / 12], dim=-1))
            pooled = (token * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
            h = self.pool(pooled)
        else:
            sequences = []
            lengths = []
            for i in range(len(x)):
                loc = torch.nonzero(mask[i]).flatten()
                if shuffle and len(loc) > 1:
                    loc = loc[torch.randperm(len(loc), generator=generator)]
                lengths.append(max(len(loc), 1))
                if len(loc) == 0:
                    sequences.append(torch.zeros(1, self.gru.input_size, device=x.device))
                    continue
                features = [x[i, loc], age[i, loc, None] / 12]
                if self.variant == "T4":
                    gaps = torch.zeros(len(loc), 1, device=x.device)
                    if len(loc) > 1:
                        gaps[1:, 0] = (age[i, loc[:-1]] - age[i, loc[1:]]).abs() / 12
                    features.append(gaps)
                sequences.append(torch.cat(features, dim=1))
            padded = pad_sequence(sequences, batch_first=True)
            _, hidden = self.gru(pack_padded_sequence(padded, lengths, batch_first=True,
                                                        enforce_sorted=False))
            h = self.project(hidden[-1])
        return h, self.head(h)


def fit_history(months_path: str, apps_path: str, record_model_dir: str,
                earliest_application: str, out_dir: str, *, variant: str,
                history_months: int = 12, out_dim: int = 16, epochs: int = 40,
                batch_size: int = 256, seed: int = 17) -> dict:
    if variant not in {"T0", "T1", "T2", "T3", "T4"}:
        raise ValueError("Unknown temporal variant")
    torch.manual_seed(seed)
    apps, x, mask, age, cols = sequence_arrays(months_path, apps_path, history_months)
    record_manifest = json.loads((Path(record_model_dir) / "manifest.json").read_text(encoding="utf-8"))
    if apps["application_time"].max() >= pd.Timestamp(earliest_application):
        raise ValueError("History encoder fit corpus crosses earliest application")
    if mask.sum(1).max(initial=0) < 2:
        raise ValueError("Need at least one history with two observed months")
    months = pd.read_parquet(months_path)
    if months["record_model_version"].nunique() != 1 or months["record_model_version"].iloc[0] != record_manifest["version"]:
        raise ValueError("Monthly vectors do not match record encoder")
    cfg = {"variant": variant, "month_dim": len(cols), "out_dim": out_dim,
           "history_months": history_months}
    model = HistoryEncoder(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    candidates = np.flatnonzero(mask.sum(1) >= 2)
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(candidates)
    validation_count = max(1, int(np.ceil(len(permuted) * 0.1)))
    if len(permuted) - validation_count < 2:
        raise ValueError("Too few history fit applications after held-out split")
    validation_indices = permuted[:validation_count]
    train_indices = permuted[validation_count:]
    latest = np.argmax(np.where(mask, np.arange(history_months)[None, :], -1), axis=1)
    target = x[np.arange(len(x)), latest].copy()
    masked = mask.copy()
    masked[np.arange(len(x)), latest] = False
    losses = []
    heldout_losses = []
    best_loss = float("inf")
    best_state = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(train_indices)
        batch_losses = []
        model.train()
        for chunk in np.array_split(order, max(1, int(np.ceil(len(order) / batch_size)))):
            xb = torch.from_numpy(x[chunk])
            mb = torch.from_numpy(masked[chunk])
            ab = torch.from_numpy(age[chunk])
            yb = torch.from_numpy(target[chunk])
            _, pred = model(xb, mb, ab, shuffle=variant == "T3")
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        losses.append(float(np.mean(batch_losses)))
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(x[validation_indices]),
                             torch.from_numpy(masked[validation_indices]),
                             torch.from_numpy(age[validation_indices]),
                             shuffle=variant == "T3",
                             generator=torch.Generator().manual_seed(seed + epoch))[1]
            validation_loss = float(nn.functional.mse_loss(
                val_pred, torch.from_numpy(target[validation_indices])).item())
        heldout_losses.append(validation_loss)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best_state)
    payload = {"kind": "history", "model_config": cfg, "record_model_version": record_manifest["version"],
               "pit_basis": record_manifest["pit_basis"], "fit_max_anchor": pd.to_datetime(apps["application_time"]).max().isoformat(),
               "seed": seed, "max_epochs": epochs, "epochs_trained": len(losses),
               "fit_applications": int(len(train_indices)), "heldout_applications": int(len(validation_indices)),
               "final_prediction_loss": losses[-1], "heldout_prediction_loss": best_loss,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "training_seconds": time.perf_counter() - started}
    return _save(model, payload, out_dir)


def encode_applications(months_path: str, apps_path: str, history_model_dir: str,
                        out_path: str, *, batch_size: int = 512, shuffle_averages: int = 5,
                        for_fit: bool = False) -> dict:
    started = time.perf_counter()
    model, manifest = _load(history_model_dir, lambda d: HistoryEncoder(d["model_config"]))
    cfg = manifest["model_config"]
    apps, x, mask, age, _ = sequence_arrays(months_path, apps_path, cfg["history_months"])
    months = pd.read_parquet(months_path)
    if len(months) and (months["record_model_version"].nunique() != 1 or
                        months["record_model_version"].iloc[0] != manifest["record_model_version"]):
        raise ValueError("Record encoder version mismatch")
    if apps["pit_basis"].nunique() != 1 or apps["pit_basis"].iloc[0] != manifest["pit_basis"]:
        raise ValueError("Source basis mismatch")
    if for_fit and pd.to_datetime(apps["application_time"]).max() > pd.Timestamp(manifest["fit_max_anchor"]):
        raise ValueError("Fit materialization includes later applications")
    if not for_fit and pd.to_datetime(apps["application_time"]).min() <= pd.Timestamp(manifest["fit_max_anchor"]):
        raise ValueError("History scoring precedes fit cutoff")
    valid = (mask.sum(1) >= 2) & apps["coverage_status"].eq("USABLE").to_numpy()
    vectors = np.full((len(apps), cfg["out_dim"]), np.nan, np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(apps), batch_size):
            end = min(start + batch_size, len(apps))
            xb, mb, ab = (torch.from_numpy(v[start:end]) for v in (x, mask, age))
            repeats = shuffle_averages if cfg["variant"] == "T3" else 1
            hs = [model(xb, mb, ab, shuffle=cfg["variant"] == "T3",
                        generator=torch.Generator().manual_seed(manifest["seed"] + k))[0].numpy()
                  for k in range(repeats)]
            vectors[start:end] = np.mean(hs, axis=0)
    vectors[~valid] = np.nan
    out = apps.copy()
    out["history_model_version"] = manifest["version"]
    out["record_model_version"] = manifest["record_model_version"]
    out["temporal_variant"] = cfg["variant"]
    for j in range(cfg["out_dim"]):
        out[f"h_{j:02d}"] = vectors[:, j]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return {"applications": len(out), "embeddable": int(valid.sum()),
            "history_model_version": manifest["version"],
            "encoding_seconds": time.perf_counter() - started}
