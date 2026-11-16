"""Render a fail-closed Athena extract over certified source adapters."""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime

IDENT = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
MONTH = re.compile(r"^\d{4}-\d{2}$")
PARTITION = re.compile(r"^\d{8}$")
SOURCE = "db_corp_dadoscadastrais_gerirriscobacen_sor_01.risco_bacen_scr_ii"


def _ident(value: str, *, qualified: bool = False) -> str:
    if not isinstance(value, str) or "REPLACE" in value:
        raise ValueError(f"Unconfigured SQL identifier: {value!r}")
    parts = value.split(".") if qualified else [value]
    if len(parts) > 2 or not all(IDENT.fullmatch(part) for part in parts):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return ".".join(f'"{part}"' for part in parts)


def _col(adapter: dict, key: str, alias: str) -> str:
    return f"{alias}.{_ident(adapter[key])}"


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        cfg = json.load(stream)
    mode = cfg.get("source_mode", "certified_ledger")
    if mode not in {"assumed_consultation", "certified_ledger"}:
        raise ValueError("Unknown source_mode")
    adapters = {"spine": ("table", "application_id", "person_id", "application_time", "product", "split", "target", "target_provenance", "customer_status")}
    if mode == "certified_ledger":
        required = ("snapshot_ledger_complete", "authorization_ledger_complete", "source_snapshot_join_audited")
        for key in required:
            if cfg.get("certification", {}).get(key) is not True:
                raise ValueError(f"Certification gate not attested: {key}")
        adapters.update({
            "snapshot": ("table", "person_id", "reference_month", "event_id", "available_at", "replaced_at", "status", "partition", "acquisition", "protocol", "consultation_raw"),
            "authorization": ("table", "person_id", "valid_from", "valid_to", "permitted_use"),
        })
    for adapter, keys in adapters.items():
        for key in keys:
            _ident(cfg[adapter][key], qualified=key == "table")
    months = cfg["application_months"]
    if len(months) != 8 or len(set(months)) != 8 or months != sorted(months) or not all(MONTH.fullmatch(x) for x in months):
        raise ValueError("application_months must contain eight distinct ordered YYYY-MM values")
    parts = cfg["source_partitions"]
    if not parts or not all(isinstance(x, str) and PARTITION.fullmatch(x) for x in parts):
        raise ValueError("source_partitions must be verified YYYYMMDD strings")
    if not isinstance(cfg["source_origin"], str) or not re.fullmatch(r"[A-Za-z0-9_]+", cfg["source_origin"]) or "REPLACE" in cfg["source_origin"]:
        raise ValueError("source_origin must be a verified single code")
    if not 1 <= cfg["minimum_history_months"] <= cfg["history_months"] <= 60:
        raise ValueError("Invalid history window")
    return cfg


def _bucket_predicate(spine: dict, bucket_count: int, bucket_index: int) -> str:
    if not 1 <= bucket_count <= 4096 or not 0 <= bucket_index < bucket_count:
        raise ValueError("Invalid application hash bucket")
    if bucket_count == 1:
        return ""
    application_id = _col(spine, "application_id", "p")
    return (f"\n   AND MOD(FROM_BASE(SUBSTR(TO_HEX(MD5(TO_UTF8(CAST({application_id} AS VARCHAR)))), 1, 8), 16), "
            f"{bucket_count}) = {bucket_index}")


def _partitioned_bucket_clause(count: int) -> str:
    if count == 0:
        return ""
    if not 1 <= count <= 100:
        raise ValueError("Athena partitioned UNLOAD supports at most 100 configured buckets")
    return (",\n       MOD(FROM_BASE(SUBSTR(TO_HEX(MD5(TO_UTF8(p.application_id))), 1, 8), 16), "
            f"{count}) bucket")


def _unload(select: str, output_s3: str, partitioned_buckets: int) -> str:
    properties = "format = 'PARQUET', compression = 'SNAPPY'"
    if partitioned_buckets:
        properties += ", partitioned_by = ARRAY['bucket']"
    return f"UNLOAD ({select})\nTO '{output_s3}'\nWITH ({properties})"


def render_sql(cfg: dict, month: str, output_s3: str,
               bucket_count: int = 1, bucket_index: int = 0,
               partitioned_buckets: int = 0) -> str:
    if month not in cfg["application_months"]:
        raise ValueError("Month is outside the configured eight-month population")
    if not re.fullmatch(r"s3://[A-Za-z0-9._-]+/[A-Za-z0-9/_=.+-]+/", output_s3):
        raise ValueError("Use a fresh, explicit S3 output prefix ending in /")
    if partitioned_buckets and (bucket_count != 1 or bucket_index != 0):
        raise ValueError("Use either a single filtered bucket or partitioned output")
    bucket_filter = _bucket_predicate(cfg["spine"], bucket_count, bucket_index)
    bucket_select = _partitioned_bucket_clause(partitioned_buckets)
    if cfg.get("source_mode") == "assumed_consultation":
        return _render_assumed_sql(cfg, month, output_s3, bucket_filter=bucket_filter,
                                   partitioned_buckets=partitioned_buckets)
    s, v, a = (cfg[x] for x in ("spine", "snapshot", "authorization"))
    sc = lambda key: _col(s, key, "p")
    vc = lambda key: _col(v, key, "v")
    ac = lambda key: _col(a, key, "a")
    partitions = ", ".join(f"'{x}'" for x in cfg["source_partitions"])
    expected_split = "TRAIN" if cfg["application_months"].index(month) < 5 else "OOT"
    consultation_ts = (
        f"COALESCE(TRY(DATE_PARSE(CAST({vc('consultation_raw')} AS VARCHAR), '%Y-%m-%d %H:%i:%s')), "
        f"TRY(DATE_PARSE(CAST({vc('consultation_raw')} AS VARCHAR), '%d.%m.%Y%H:%i:%s')), "
        f"TRY(DATE_PARSE(CAST({vc('consultation_raw')} AS VARCHAR), '%d.%m.%Y %H:%i:%s')))"
    )
    # Source columns below were observed in Stage 0.1; snapshot and authorization
    # columns are intentionally configured, never guessed.
    select = f"""
WITH spine_base AS (
 SELECT CAST({sc('application_id')} AS VARCHAR) application_id,
        CAST({sc('person_id')} AS VARCHAR) person_id,
        CAST({sc('application_time')} AS TIMESTAMP) application_time,
        CAST({sc('product')} AS VARCHAR) product,
        CAST({sc('split')} AS VARCHAR) split,
        {sc('target')} target_value,
        CAST({sc('target_provenance')} AS VARCHAR) target_provenance,
        CAST({sc('customer_status')} AS VARCHAR) customer_status,
        '{expected_split}' expected_split
 FROM {_ident(s['table'], qualified=True)} p
 WHERE CAST({sc('application_time')} AS DATE) >= DATE '{month}-01'
   AND CAST({sc('application_time')} AS DATE) < DATE_ADD('month', 1, DATE '{month}-01'){bucket_filter}
), spine AS (
 SELECT *, COUNT(*) OVER (PARTITION BY application_id, product) spine_duplicate_count
 FROM spine_base
), auth_rollup AS (
 SELECT p.application_id, p.product,
        COUNT({ac('person_id')}) auth_intervals,
        COUNT_IF({ac('permitted_use')} = TRUE) permitted_intervals
 FROM spine p LEFT JOIN {_ident(a['table'], qualified=True)} a
   ON p.person_id = CAST({ac('person_id')} AS VARCHAR)
  AND CAST({ac('valid_from')} AS TIMESTAMP) <= p.application_time
  AND ({ac('valid_to')} IS NULL OR CAST({ac('valid_to')} AS TIMESTAMP) > p.application_time)
 GROUP BY p.application_id, p.product
), eligible_spine AS (
 SELECT p.* FROM spine p JOIN auth_rollup a
 ON p.application_id = a.application_id AND p.product = a.product
 WHERE a.auth_intervals = 1 AND a.permitted_intervals = 1
), candidates AS (
 SELECT p.application_id, p.product,
        CAST({vc('event_id')} AS VARCHAR) event_id,
        CAST({vc('reference_month')} AS VARCHAR) scr_ref_month,
        CAST({vc('available_at')} AS TIMESTAMP) available_at,
        CAST({vc('replaced_at')} AS TIMESTAMP) replaced_at,
        CAST({vc('status')} AS VARCHAR) snapshot_status,
        CAST({vc('partition')} AS VARCHAR) source_partition,
        CAST({vc('acquisition')} AS VARCHAR) acquisition,
        CAST({vc('protocol')} AS VARCHAR) protocol,
        CAST({vc('consultation_raw')} AS VARCHAR) consultation_raw
 FROM eligible_spine p JOIN {_ident(v['table'], qualified=True)} v
   ON p.person_id = CAST({vc('person_id')} AS VARCHAR)
  AND CAST({vc('available_at')} AS TIMESTAMP) < p.application_time
  AND ({vc('replaced_at')} IS NULL OR CAST({vc('replaced_at')} AS TIMESTAMP) > p.application_time)
  AND CAST(TRY(DATE_PARSE(CAST({vc('reference_month')} AS VARCHAR), '%Y%m')) AS DATE)
      >= DATE_ADD('month', -{cfg['history_months']}, DATE_TRUNC('month', p.application_time))
  AND CAST(TRY(DATE_PARSE(CAST({vc('reference_month')} AS VARCHAR), '%Y%m')) AS DATE)
      < DATE_TRUNC('month', p.application_time)
  AND {consultation_ts} <= CAST({vc('available_at')} AS TIMESTAMP)
  AND {consultation_ts} < p.application_time
 WHERE CAST({vc('partition')} AS VARCHAR) IN ({partitions})
   AND {vc('status')} IN ('REPORT', 'NO_REPORT', 'NO_EXPOSURE')
), ranked AS (
 SELECT c.*,
        ROW_NUMBER() OVER (PARTITION BY application_id, product, scr_ref_month
                           ORDER BY available_at DESC, event_id DESC) version_rank,
        COUNT(*) OVER (PARTITION BY application_id, product, scr_ref_month, available_at) same_time_events
 FROM candidates c
), selected AS (
 SELECT * FROM ranked WHERE version_rank = 1
), coverage AS (
 SELECT application_id, product,
        COUNT_IF(snapshot_status = 'REPORT') report_months,
        COUNT_IF(snapshot_status = 'NO_REPORT') no_report_months,
        COUNT_IF(snapshot_status = 'NO_EXPOSURE') no_exposure_months,
        MAX(same_time_events) max_same_time_events
 FROM selected GROUP BY application_id, product
), source_rows AS (
 SELECT CAST(cod_idef_pess AS VARCHAR) person_id,
        CAST(txt_ano_mes_risc_bace AS VARCHAR) scr_ref_month,
        CAST(anomesdia AS VARCHAR) source_partition,
        CAST(cod_apla_risc_bace AS VARCHAR) acquisition,
        CAST(cod_prco_risc_bace AS VARCHAR) protocol,
        CAST(dat_hor_cslt_risc_bace AS VARCHAR) consultation_raw,
        CAST(cod_chav_risc_bace AS VARCHAR) record_key,
        CAST(cod_moda_cred_risc_bace AS VARCHAR) modality,
        CAST(cod_orig_risc_bace AS VARCHAR) origin,
        CAST(cod_inst_finn_risc_bace AS VARCHAR) institution,
        CAST(vlr_totl_cred_vncr_risc AS DOUBLE) balance_current,
        CAST(vlr_totl_cred_vncd_risc AS DOUBLE) balance_overdue
 FROM {SOURCE}
 WHERE anomesdia IN ({partitions})
   AND cod_orig_risc_bace = '{cfg['source_origin']}'
   AND cod_idef_pess IN (SELECT DISTINCT person_id FROM eligible_spine)
), app_coverage AS (
 SELECT p.*, a.auth_intervals, a.permitted_intervals,
        COALESCE(c.report_months, 0) report_months,
        COALESCE(c.no_report_months, 0) no_report_months,
        COALESCE(c.no_exposure_months, 0) no_exposure_months,
        CASE WHEN a.auth_intervals > 1 THEN 'AMBIGUOUS_AUTHORIZATION'
             WHEN a.auth_intervals = 0 OR a.permitted_intervals = 0 THEN 'NO_AUTHORIZATION'
             WHEN c.max_same_time_events > 1 THEN 'AMBIGUOUS_VERSION'
             WHEN c.report_months >= {cfg['minimum_history_months']} THEN 'USABLE'
             WHEN c.report_months > 0 THEN 'INSUFFICIENT_HISTORY'
             WHEN c.no_exposure_months > 0 THEN 'NO_EXPOSURE'
             WHEN c.no_report_months > 0 THEN 'NO_REPORT'
             ELSE 'UNKNOWN_AVAILABILITY' END coverage_status
 FROM spine p JOIN auth_rollup a
   ON p.application_id = a.application_id AND p.product = a.product
 LEFT JOIN coverage c
   ON p.application_id = c.application_id AND p.product = c.product
)
SELECT p.application_id, p.person_id, p.application_time, p.product, p.split,
       p.expected_split, p.spine_duplicate_count,
       'CERTIFIED_LEDGER' pit_basis,
       CASE WHEN p.auth_intervals = 1 AND p.permitted_intervals = 1 THEN 'AUTHORIZED'
            WHEN p.auth_intervals = 0 OR p.permitted_intervals = 0 THEN 'NO_AUTHORIZATION'
            ELSE 'AMBIGUOUS' END authorization_status,
       p.target_value, p.target_provenance, p.customer_status, p.coverage_status,
       p.report_months, p.no_report_months, p.no_exposure_months,
       s.event_id, s.scr_ref_month, s.available_at, s.replaced_at,
       s.snapshot_status, s.source_partition, s.acquisition, s.protocol,
       s.consultation_raw, r.record_key, r.modality, r.origin, r.institution,
       r.balance_current, r.balance_overdue{bucket_select}
FROM app_coverage p
LEFT JOIN selected s
  ON p.application_id = s.application_id AND p.product = s.product
LEFT JOIN source_rows r
  ON s.snapshot_status = 'REPORT'
 AND p.person_id = r.person_id AND s.scr_ref_month = r.scr_ref_month
 AND s.source_partition = r.source_partition
 AND s.acquisition = r.acquisition AND s.protocol = r.protocol
 AND s.consultation_raw = r.consultation_raw
""".strip()
    return _unload(select, output_s3, partitioned_buckets)


def render_fit_sql(cfg: dict, anchor: str, earliest_application: str,
                   output_s3: str, sample_mod: int = 1000) -> str:
    """Build an unlabeled pre-cutoff SCR corpus without future applicant membership."""
    if cfg.get("source_mode") != "assumed_consultation":
        raise ValueError("Direct SCR fit extraction currently supports assumed_consultation mode")
    try:
        anchor_time = datetime.fromisoformat(anchor)
        first_time = datetime.fromisoformat(earliest_application)
    except ValueError as exc:
        raise ValueError("Use ISO local timestamps for anchor and earliest application") from exc
    if not anchor_time < first_time:
        raise ValueError("Fit anchor must be strictly before earliest application")
    if not 1 <= sample_mod <= 100000:
        raise ValueError("sample_mod outside supported range")
    if not re.fullmatch(r"s3://[A-Za-z0-9._-]+/[A-Za-z0-9/_=.+-]+/", output_s3):
        raise ValueError("Use a fresh, explicit S3 output prefix ending in /")
    return _render_assumed_sql(cfg, cfg["application_months"][0], output_s3,
                               fit_anchor=anchor_time.strftime("%Y-%m-%d %H:%M:%S"),
                               sample_mod=sample_mod)


def render_anchor_sql(cfg: dict, month: str, offset_months: int, fit_cutoff: str,
                      output_s3: str, bucket_count: int = 1,
                      bucket_index: int = 0, partitioned_buckets: int = 0) -> str:
    """Re-extract earlier as-of states of the original application audience."""
    if cfg.get("source_mode") != "assumed_consultation":
        raise ValueError("Anchor extraction currently supports assumed_consultation mode")
    if month not in cfg["application_months"] or not 1 <= offset_months <= cfg["history_months"]:
        raise ValueError("Invalid original month or anchor offset")
    try:
        cutoff = datetime.fromisoformat(fit_cutoff).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError("Use an ISO fit cutoff") from exc
    if not re.fullmatch(r"s3://[A-Za-z0-9._-]+/[A-Za-z0-9/_=.+-]+/", output_s3):
        raise ValueError("Use a fresh, explicit S3 output prefix ending in /")
    if partitioned_buckets and (bucket_count != 1 or bucket_index != 0):
        raise ValueError("Use either a single filtered bucket or partitioned output")
    bucket_filter = _bucket_predicate(cfg["spine"], bucket_count, bucket_index)
    return _render_assumed_sql(cfg, month, output_s3, anchor_offset=offset_months,
                               anchor_fit_cutoff=cutoff, bucket_filter=bucket_filter,
                               partitioned_buckets=partitioned_buckets)


def _render_assumed_sql(cfg: dict, month: str, output_s3: str,
                        fit_anchor: str | None = None, sample_mod: int = 1000,
                        anchor_offset: int | None = None,
                        anchor_fit_cutoff: str | None = None,
                        bucket_filter: str = "", partitioned_buckets: int = 0) -> str:
    """Development adapter using only Stage 0.1 observed physical columns.

    Assumption: consultation time is the first bank-usable time for the retained
    detailed record. No authorization or replacement history is asserted here.
    """
    s = cfg["spine"]
    sc = lambda key: _col(s, key, "p")
    partitions = ", ".join(f"'{x}'" for x in cfg["source_partitions"])
    bucket_select = _partitioned_bucket_clause(partitioned_buckets)
    expected_split = "TRAIN" if cfg["application_months"].index(month) < 5 else "OOT"
    if anchor_offset is not None:
        anchor_expr = f"DATE_ADD('month', -{anchor_offset}, CAST({sc('application_time')} AS TIMESTAMP))"
        spine_sql = f"""
 SELECT CONCAT(CAST({sc('application_id')} AS VARCHAR), '__anchor_{anchor_offset}') application_id,
        CAST({sc('application_id')} AS VARCHAR) parent_application_id,
        CAST({sc('person_id')} AS VARCHAR) person_id,
        {anchor_expr} application_time,
        {anchor_expr} anchor_time,
        CAST({sc('product')} AS VARCHAR) product,
        CAST({sc('split')} AS VARCHAR) split,
        CAST(NULL AS DOUBLE) target_value,
        'UNLABELED' target_provenance,
        CAST({sc('customer_status')} AS VARCHAR) customer_status,
        '{expected_split}' expected_split
 FROM {_ident(s['table'], qualified=True)} p
 WHERE CAST({sc('application_time')} AS DATE) >= DATE '{month}-01'
   AND CAST({sc('application_time')} AS DATE) < DATE_ADD('month', 1, DATE '{month}-01'){bucket_filter}
   AND {anchor_expr} > TIMESTAMP '{anchor_fit_cutoff}'
"""
    elif fit_anchor is None:
        spine_sql = f"""
 SELECT CAST({sc('application_id')} AS VARCHAR) application_id,
        CAST(NULL AS VARCHAR) parent_application_id,
        CAST({sc('person_id')} AS VARCHAR) person_id,
        CAST({sc('application_time')} AS TIMESTAMP) application_time,
        CAST(NULL AS TIMESTAMP) anchor_time,
        CAST({sc('product')} AS VARCHAR) product,
        CAST({sc('split')} AS VARCHAR) split,
        {sc('target')} target_value,
        CAST({sc('target_provenance')} AS VARCHAR) target_provenance,
        CAST({sc('customer_status')} AS VARCHAR) customer_status,
        '{expected_split}' expected_split
 FROM {_ident(s['table'], qualified=True)} p
 WHERE CAST({sc('application_time')} AS DATE) >= DATE '{month}-01'
   AND CAST({sc('application_time')} AS DATE) < DATE_ADD('month', 1, DATE '{month}-01'){bucket_filter}
"""
    else:
        parse_consult = """COALESCE(
          TRY(DATE_PARSE(dat_hor_cslt_risc_bace, '%Y-%m-%d %H:%i:%s')),
          TRY(DATE_PARSE(dat_hor_cslt_risc_bace, '%d.%m.%Y%H:%i:%s')),
          TRY(DATE_PARSE(dat_hor_cslt_risc_bace, '%d.%m.%Y %H:%i:%s')))"""
        spine_sql = f"""
 SELECT CONCAT('FIT_', TO_HEX(MD5(TO_UTF8(CAST(cod_idef_pess AS VARCHAR))))) application_id,
        CAST(NULL AS VARCHAR) parent_application_id,
        CAST(cod_idef_pess AS VARCHAR) person_id,
        TIMESTAMP '{fit_anchor}' application_time,
        CAST(NULL AS TIMESTAMP) anchor_time,
        'PRETRAIN' product, 'FIT' split, CAST(NULL AS DOUBLE) target_value,
        'UNLABELED' target_provenance, 'UNKNOWN' customer_status,
        'FIT' expected_split
 FROM {SOURCE}
 WHERE anomesdia IN ({partitions})
   AND cod_idef_pess IS NOT NULL
   AND {parse_consult} < TIMESTAMP '{fit_anchor}'
   AND MOD(FROM_BASE(SUBSTR(TO_HEX(MD5(TO_UTF8(CAST(cod_idef_pess AS VARCHAR)))), 1, 8), 16), {sample_mod}) = 0
 GROUP BY cod_idef_pess
"""
    sql = f"""
WITH spine_base AS (
{spine_sql}
), spine AS (
 SELECT *, COUNT(*) OVER (PARTITION BY application_id, product) spine_duplicate_count
 FROM spine_base
), source_parsed AS (
 SELECT CAST(cod_idef_pess AS VARCHAR) person_id,
        CAST(txt_ano_mes_risc_bace AS VARCHAR) scr_ref_month,
        CAST(anomesdia AS VARCHAR) source_partition,
        CAST(cod_apla_risc_bace AS VARCHAR) acquisition,
        CAST(cod_prco_risc_bace AS VARCHAR) protocol,
        CAST(dat_hor_cslt_risc_bace AS VARCHAR) consultation_raw,
        COALESCE(
          TRY(DATE_PARSE(dat_hor_cslt_risc_bace, '%Y-%m-%d %H:%i:%s')),
          TRY(DATE_PARSE(dat_hor_cslt_risc_bace, '%d.%m.%Y%H:%i:%s')),
          TRY(DATE_PARSE(dat_hor_cslt_risc_bace, '%d.%m.%Y %H:%i:%s'))
        ) consultation_ts,
        CAST(cod_chav_risc_bace AS VARCHAR) record_key,
        CAST(cod_moda_cred_risc_bace AS VARCHAR) modality,
        CAST(cod_orig_risc_bace AS VARCHAR) origin,
        CAST(cod_inst_finn_risc_bace AS VARCHAR) institution,
        CAST(vlr_totl_cred_vncr_risc AS DOUBLE) balance_current,
        CAST(vlr_totl_cred_vncd_risc AS DOUBLE) balance_overdue
 FROM {SOURCE}
 WHERE anomesdia IN ({partitions})
   AND cod_orig_risc_bace = '{cfg['source_origin']}'
   AND cod_idef_pess IN (SELECT DISTINCT person_id FROM spine)
   AND ((cod_apla_risc_bace = 'Consulta Lote'
         AND TRY_CAST(cod_prco_risc_bace AS BIGINT) IS NOT NULL)
        OR (cod_apla_risc_bace = 'Consulta Online'
            AND cod_prco_risc_bace = 'Externo'))
), source_rows AS (
 SELECT * FROM source_parsed
 WHERE consultation_ts IS NOT NULL
   AND TRY(DATE_PARSE(scr_ref_month, '%Y%m')) IS NOT NULL
), snapshots AS (
 SELECT person_id, scr_ref_month, source_partition, acquisition, protocol,
        consultation_raw, consultation_ts available_at,
        TO_HEX(MD5(TO_UTF8(CONCAT(
          person_id, '|', scr_ref_month, '|', source_partition, '|', acquisition,
          '|', protocol, '|', consultation_raw)))) event_id,
        'REPORT' snapshot_status
 FROM source_rows
 GROUP BY person_id, scr_ref_month, source_partition, acquisition,
          protocol, consultation_raw, consultation_ts
), candidates AS (
 SELECT p.application_id, p.product, v.*
 FROM spine p JOIN snapshots v ON p.person_id = v.person_id
  AND v.available_at < p.application_time
  AND CAST(TRY(DATE_PARSE(v.scr_ref_month, '%Y%m')) AS DATE)
      >= DATE_ADD('month', -{cfg['history_months']}, DATE_TRUNC('month', p.application_time))
  AND CAST(TRY(DATE_PARSE(v.scr_ref_month, '%Y%m')) AS DATE)
      < DATE_TRUNC('month', p.application_time)
), ranked AS (
 SELECT c.*,
        ROW_NUMBER() OVER (PARTITION BY application_id, product, scr_ref_month
                           ORDER BY available_at DESC, event_id DESC) version_rank,
        COUNT(*) OVER (PARTITION BY application_id, product, scr_ref_month, available_at)
          same_time_events
 FROM candidates c
), selected AS (SELECT * FROM ranked WHERE version_rank = 1),
coverage AS (
 SELECT application_id, product, COUNT(*) report_months,
        MAX(same_time_events) max_same_time_events
 FROM selected GROUP BY application_id, product
), app_coverage AS (
 SELECT p.*,
        COALESCE(c.report_months, 0) report_months,
        CAST(0 AS BIGINT) no_report_months,
        CAST(0 AS BIGINT) no_exposure_months,
        CASE WHEN c.max_same_time_events > 1 THEN 'AMBIGUOUS_VERSION'
             WHEN c.report_months >= {cfg['minimum_history_months']} THEN 'USABLE'
             WHEN c.report_months > 0 THEN 'INSUFFICIENT_HISTORY'
             ELSE 'UNKNOWN_AVAILABILITY' END coverage_status
 FROM spine p LEFT JOIN coverage c
   ON p.application_id = c.application_id AND p.product = c.product
)
SELECT p.application_id, p.parent_application_id, p.person_id,
       p.application_time, p.anchor_time, p.product, p.split,
       p.expected_split, p.spine_duplicate_count,
       'ASSUMED_CONSULTATION' pit_basis,
       CASE WHEN p.report_months > 0 THEN 'ASSUMED_FROM_SCR_EVENT'
            ELSE 'UNKNOWN' END authorization_status,
       p.target_value, p.target_provenance, p.customer_status, p.coverage_status,
       p.report_months, p.no_report_months, p.no_exposure_months,
       s.event_id, s.scr_ref_month, s.available_at,
       CAST(NULL AS TIMESTAMP) replaced_at,
       s.snapshot_status, s.source_partition, s.acquisition, s.protocol,
       s.consultation_raw, r.record_key, r.modality, r.origin, r.institution,
       r.balance_current, r.balance_overdue{bucket_select}
FROM app_coverage p
LEFT JOIN selected s
  ON p.application_id = s.application_id AND p.product = s.product
LEFT JOIN source_rows r
  ON p.person_id = r.person_id AND s.scr_ref_month = r.scr_ref_month
 AND s.source_partition = r.source_partition
 AND s.acquisition = r.acquisition AND s.protocol = r.protocol
 AND s.consultation_raw = r.consultation_raw
""".strip()
    return _unload(sql, output_s3, partitioned_buckets)
