import logging
from typing import Optional, Union

import pandas as pd
import polars as pl

from core.snapshot import hash_file, hash_dataframe, snapshot_file, snapshot_dataframe
from db.models import DatasetVersion
from store.blob import BlobStore
from store.metadata import MetadataStore

logger = logging.getLogger(__name__)


class DatasetTracker:
    def __init__(self, metadata_store: MetadataStore, blob_store: BlobStore):
        self._meta = metadata_store
        self._blob = blob_store

    async def track_file(
        self,
        name: str,
        path: str,
        source_type: str = "csv",
        description: str = "",
        tags: Optional[list] = None,
        created_by: str = "system",
    ) -> DatasetVersion:
        content_hash, _ = hash_file(path)

        existing = await self._meta.get_version_by_hash(content_hash)
        if existing and existing.name == name:
            logger.info("Dataset %s: identical content already tracked as v%d", name, existing.version)
            return existing

        snap = await snapshot_file(name, path, self._blob)
        return await self._meta.create_dataset_version(
            name=name,
            content_hash=snap["content_hash"],
            source_path=path,
            source_type=source_type,
            row_count=snap["row_count"],
            column_count=snap["column_count"],
            schema_json=snap["schema_json"],
            storage_path=snap["storage_path"],
            created_by=created_by,
            description=description or None,
            tags=tags or [],
        )

    async def track_dataframe(
        self,
        name: str,
        df: Union[pd.DataFrame, pl.DataFrame],
        source_type: str = "dataframe",
        description: str = "",
        tags: Optional[list] = None,
        created_by: str = "system",
    ) -> DatasetVersion:
        content_hash, _ = hash_dataframe(df)

        existing = await self._meta.get_version_by_hash(content_hash)
        if existing and existing.name == name:
            logger.info("Dataset %s: identical content already tracked as v%d", name, existing.version)
            return existing

        snap = await snapshot_dataframe(name, df, self._blob)
        return await self._meta.create_dataset_version(
            name=name,
            content_hash=snap["content_hash"],
            source_path="<dataframe>",
            source_type=source_type,
            row_count=snap["row_count"],
            column_count=snap["column_count"],
            schema_json=snap["schema_json"],
            storage_path=snap["storage_path"],
            created_by=created_by,
            description=description or None,
            tags=tags or [],
        )

    async def get_version(self, name: str, version: Optional[int] = None) -> Optional[DatasetVersion]:
        if version is None:
            return await self._meta.get_latest_version(name)
        versions = await self._meta.get_all_versions(name)
        return next((v for v in versions if v.version == version), None)

    async def list_versions(self, name: str) -> list[DatasetVersion]:
        return await self._meta.get_all_versions(name)

    async def diff(self, name: str, version_a: int, version_b: int) -> dict:
        va = await self.get_version(name, version_a)
        vb = await self.get_version(name, version_b)

        if va is None or vb is None:
            raise ValueError(f"One or both versions not found: v{version_a}, v{version_b}")

        schema_a: dict = va.schema_json or {}
        schema_b: dict = vb.schema_json or {}

        added = {c: schema_b[c] for c in schema_b if c not in schema_a}
        removed = {c: schema_a[c] for c in schema_a if c not in schema_b}
        changed = {
            c: {"from": schema_a[c], "to": schema_b[c]}
            for c in schema_a
            if c in schema_b and schema_a[c] != schema_b[c]
        }

        return {
            "schema_changes": {"added": added, "removed": removed, "changed": changed},
            "row_count_delta": vb.row_count - va.row_count,
            "column_count_delta": vb.column_count - va.column_count,
            "hash_changed": va.content_hash != vb.content_hash,
            "content_identical": va.content_hash == vb.content_hash,
            "version_a_stats": {
                "version": va.version,
                "row_count": va.row_count,
                "column_count": va.column_count,
                "schema": schema_a,
                "created_at": va.created_at.isoformat(),
            },
            "version_b_stats": {
                "version": vb.version,
                "row_count": vb.row_count,
                "column_count": vb.column_count,
                "schema": schema_b,
                "created_at": vb.created_at.isoformat(),
            },
        }

    async def load_version(self, name: str, version: int) -> bytes:
        dv = await self.get_version(name, version)
        if dv is None:
            raise ValueError(f"Dataset {name} v{version} not found")
        return await self._blob.async_download_bytes(dv.storage_path)
