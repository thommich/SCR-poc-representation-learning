"""Optional AWS execution and Parquet retrieval for rendered Athena UNLOAD SQL."""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlparse


def _client(service: str, region: str | None = None):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install scr-repr-poc[aws] to execute Athena queries") from exc
    return boto3.client(service, region_name=region)


def execute_unload(sql: str, *, workgroup: str, query_result_s3: str,
                   region: str | None = None, timeout_seconds: int = 3600) -> str:
    if not sql.lstrip().upper().startswith("UNLOAD ("):
        raise ValueError("Only rendered UNLOAD statements are accepted")
    athena = _client("athena", region)
    execution_id = athena.start_query_execution(
        QueryString=sql,
        WorkGroup=workgroup,
        ResultConfiguration={"OutputLocation": query_result_s3},
    )["QueryExecutionId"]
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = athena.get_query_execution(QueryExecutionId=execution_id)["QueryExecution"]["Status"]
        state = status["State"]
        if state == "SUCCEEDED":
            return execution_id
        if state in {"FAILED", "CANCELLED"}:
            raise RuntimeError(f"Athena {state}: {status.get('StateChangeReason', 'no reason returned')}")
        time.sleep(5)
    raise TimeoutError(f"Athena query {execution_id} still running after {timeout_seconds}s")


def download_parquet_prefix(s3_prefix: str, local_dir: str, *,
                            region: str | None = None) -> list[str]:
    parsed = urlparse(s3_prefix)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.endswith("/"):
        raise ValueError("Expected s3://bucket/prefix/")
    s3 = _client("s3", region)
    prefix = parsed.path.lstrip("/")
    root = Path(local_dir)
    root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=parsed.netloc, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith(".parquet"):
                continue
            relative = Path(key[len(prefix):])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unexpected S3 object key outside requested prefix")
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(parsed.netloc, key, str(target))
            downloaded.append(str(target))
    if not downloaded:
        raise ValueError("No Parquet files under Athena output prefix")
    return downloaded
