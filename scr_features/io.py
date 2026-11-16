"""Streaming Parquet concatenation for bounded application buckets."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def combine_parquet(paths: list[str], output: str, batch_size: int = 65536) -> str:
    if not paths:
        raise ValueError("No Parquet inputs")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for path in paths:
            source = pq.ParquetFile(path)
            for batch in source.iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(target, table.schema, compression="snappy")
                else:
                    table = table.cast(writer.schema)
                writer.write_table(table)
        if writer is None:
            raise ValueError("All Parquet inputs were empty")
    finally:
        if writer is not None:
            writer.close()
    return str(target)
