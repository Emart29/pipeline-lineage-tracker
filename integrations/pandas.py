import functools
from typing import Callable, Optional

import pandas as pd

from core.dataset import DatasetTracker
from core.pipeline import PipelineTracker
from db.models import DatasetVersion, PipelineRun


def track_dataframe(
    name: str,
    tracker: DatasetTracker,
    pipeline_tracker: Optional[PipelineTracker] = None,
    pipeline_run: Optional[PipelineRun] = None,
    tags: Optional[list] = None,
):
    """Decorator that auto-tracks the returned DataFrame after a function call."""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            result = await fn(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                version = await tracker.track_dataframe(name, result, tags=tags or [])
                if pipeline_tracker and pipeline_run:
                    await pipeline_tracker.link_output(pipeline_run.id, version)
            return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            import asyncio
            result = fn(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                loop = asyncio.get_event_loop()
                version = loop.run_until_complete(tracker.track_dataframe(name, result, tags=tags or []))
                if pipeline_tracker and pipeline_run:
                    loop.run_until_complete(pipeline_tracker.link_output(pipeline_run.id, version))
            return result

        import asyncio
        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


class tracked_dataframe_ctx:
    """Context manager for inline DataFrame tracking."""

    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        tracker: DatasetTracker,
        pipeline_tracker: Optional[PipelineTracker] = None,
        pipeline_run: Optional[PipelineRun] = None,
        tags: Optional[list] = None,
    ):
        self._name = name
        self._df = df
        self._tracker = tracker
        self._pipeline_tracker = pipeline_tracker
        self._pipeline_run = pipeline_run
        self._tags = tags or []
        self._version: Optional[DatasetVersion] = None

    async def __aenter__(self) -> DatasetVersion:
        self._version = await self._tracker.track_dataframe(
            self._name, self._df, tags=self._tags
        )
        if self._pipeline_tracker and self._pipeline_run:
            await self._pipeline_tracker.link_output(self._pipeline_run.id, self._version)
        return self._version

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


async def track_read_csv(
    path: str,
    name: str,
    tracker: DatasetTracker,
    pipeline_tracker: Optional[PipelineTracker] = None,
    pipeline_run: Optional[PipelineRun] = None,
    tags: Optional[list] = None,
    **kwargs,
) -> tuple[pd.DataFrame, DatasetVersion]:
    """Read a CSV and auto-track it as a dataset version."""
    df = pd.read_csv(path, **kwargs)
    version = await tracker.track_file(name, path, source_type="csv", tags=tags or [])
    if pipeline_tracker and pipeline_run:
        await pipeline_tracker.link_input(pipeline_run.id, version)
    return df, version


def compare_dataframes(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Pure utility — compare two DataFrames for schema and value drift."""
    schema_a = {col: str(dtype) for col, dtype in df_a.dtypes.items()}
    schema_b = {col: str(dtype) for col, dtype in df_b.dtypes.items()}

    added = {c: schema_b[c] for c in schema_b if c not in schema_a}
    removed = {c: schema_a[c] for c in schema_a if c not in schema_b}
    changed = {
        c: {"from": schema_a[c], "to": schema_b[c]}
        for c in schema_a
        if c in schema_b and schema_a[c] != schema_b[c]
    }

    common_cols = [c for c in df_a.columns if c in df_b.columns]
    numeric_stats = {}
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(df_a[col]) and pd.api.types.is_numeric_dtype(df_b[col]):
            numeric_stats[col] = {
                "mean_shift": float(df_b[col].mean() - df_a[col].mean()),
                "std_shift": float(df_b[col].std() - df_a[col].std()),
            }

    new_nulls = {}
    for col in common_cols:
        null_a = df_a[col].isnull().mean()
        null_b = df_b[col].isnull().mean()
        if null_b > null_a:
            new_nulls[col] = {"before": float(null_a), "after": float(null_b)}

    return {
        "schema_diff": {"added": added, "removed": removed, "changed": changed},
        "row_count_delta": len(df_b) - len(df_a),
        "numeric_column_stats": numeric_stats,
        "new_nulls": new_nulls,
    }
