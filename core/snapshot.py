import hashlib
import io
from pathlib import Path
from typing import Union

import pandas as pd
import polars as pl

from store.blob import BlobStore


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Union[str, Path]) -> tuple[str, bytes]:
    data = Path(path).read_bytes()
    return hash_bytes(data), data


def hash_dataframe(df: Union[pd.DataFrame, pl.DataFrame]) -> tuple[str, bytes]:
    if isinstance(df, pl.DataFrame):
        buf = io.BytesIO()
        df.write_parquet(buf)
        data = buf.getvalue()
    else:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        data = buf.getvalue()
    return hash_bytes(data), data


def extract_schema(df: Union[pd.DataFrame, pl.DataFrame]) -> dict:
    if isinstance(df, pl.DataFrame):
        return {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    else:
        return {col: str(dtype) for col, dtype in df.dtypes.items()}


async def snapshot_file(name: str, path: str, blob_store: BlobStore) -> dict:
    content_hash, data = hash_file(path)
    p = Path(path)
    suffix = p.suffix or ".bin"
    key = f"{name}/snapshots/{content_hash}{suffix}"

    if not await blob_store.async_object_exists(key):
        await blob_store.async_upload_bytes(key, data, "application/octet-stream")

    # Attempt row count via pandas for CSV/parquet
    row_count = 0
    schema_json = {}
    try:
        if suffix == ".csv":
            preview = pd.read_csv(path, nrows=0)
            schema_json = {col: str(dtype) for col, dtype in preview.dtypes.items()}
            row_count = sum(1 for _ in open(path)) - 1  # rough count minus header
        elif suffix in (".parquet", ".pq"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            row_count = pf.metadata.num_rows
            schema_json = {f.name: str(f.physical_type) for f in pf.schema_arrow}
    except Exception:
        pass

    return {
        "content_hash": content_hash,
        "storage_path": key,
        "size_bytes": len(data),
        "row_count": row_count,
        "column_count": len(schema_json),
        "schema_json": schema_json,
    }


async def snapshot_dataframe(name: str, df: Union[pd.DataFrame, pl.DataFrame], blob_store: BlobStore) -> dict:
    content_hash, data = hash_dataframe(df)
    key = f"{name}/snapshots/{content_hash}.parquet"

    if not await blob_store.async_object_exists(key):
        await blob_store.async_upload_bytes(key, data, "application/octet-stream")

    schema_json = extract_schema(df)

    if isinstance(df, pl.DataFrame):
        row_count = df.height
        column_count = df.width
    else:
        row_count = len(df)
        column_count = len(df.columns)

    return {
        "content_hash": content_hash,
        "storage_path": key,
        "size_bytes": len(data),
        "row_count": row_count,
        "column_count": column_count,
        "schema_json": schema_json,
    }
