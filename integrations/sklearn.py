import io
import uuid
import asyncio
import logging
from typing import Any, Optional

import joblib
from sklearn.pipeline import Pipeline

from core.dataset import DatasetTracker
from core.pipeline import PipelineTracker
from db.models import ModelArtifact, PipelineRun
from store.blob import BlobStore
from store.metadata import MetadataStore

logger = logging.getLogger(__name__)


class TrackedPipeline(Pipeline):
    """sklearn Pipeline wrapper that auto-records lineage on fit()."""

    def __init__(
        self,
        steps: list,
        pipeline_tracker: PipelineTracker,
        dataset_tracker: DatasetTracker,
        metadata_store: MetadataStore,
        blob_store: BlobStore,
        pipeline_name: str,
        model_name: str,
        parameters: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(steps, **kwargs)
        self._pipeline_tracker = pipeline_tracker
        self._dataset_tracker = dataset_tracker
        self._meta = metadata_store
        self._blob = blob_store
        self._pipeline_name = pipeline_name
        self._model_name = model_name
        self._parameters = parameters or {}
        self._run: Optional[PipelineRun] = None
        self._model_artifact: Optional[ModelArtifact] = None

    def fit(self, X, y=None, **fit_params):
        loop = asyncio.get_event_loop()
        self._run = loop.run_until_complete(
            self._pipeline_tracker.start_run(self._pipeline_name, self._parameters)
        )

        super().fit(X, y, **fit_params)

        try:
            score = float(self.score(X, y)) if hasattr(self, "score") and y is not None else None
        except Exception:
            score = None

        metrics = {
            "n_samples": int(X.shape[0]) if hasattr(X, "shape") else len(X),
            "n_features": int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) > 1 else 0,
        }
        if score is not None:
            metrics["train_score"] = score

        # Get next version number
        existing_versions = loop.run_until_complete(
            self._meta.get_model_by_name(self._model_name)
        )
        next_version = (existing_versions.version + 1) if existing_versions else 1

        self._model_artifact = loop.run_until_complete(
            track_sklearn_model(
                model=self,
                name=self._model_name,
                version=next_version,
                pipeline_run=self._run,
                metadata_store=self._meta,
                blob_store=self._blob,
                metrics=metrics,
                parameters=self._parameters,
            )
        )

        loop.run_until_complete(
            self._pipeline_tracker.finish_run(self._run.id, metrics=metrics)
        )

        return self

    @property
    def lineage_run(self) -> Optional[PipelineRun]:
        return self._run

    @property
    def lineage_model(self) -> Optional[ModelArtifact]:
        return self._model_artifact


async def track_sklearn_model(
    model: Any,
    name: str,
    version: int,
    pipeline_run: PipelineRun,
    metadata_store: MetadataStore,
    blob_store: BlobStore,
    metrics: Optional[dict] = None,
    parameters: Optional[dict] = None,
) -> ModelArtifact:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    model_bytes = buf.getvalue()

    storage_key = f"models/{name}/v{version}/{name}.joblib"
    await blob_store.async_upload_bytes(storage_key, model_bytes, "application/octet-stream")

    artifact = await metadata_store.create_model_artifact(
        name=name,
        version=version,
        pipeline_run_id=pipeline_run.id,
        storage_path=storage_key,
        framework="sklearn",
        metrics=metrics or {},
        parameters=parameters or {},
    )

    # Edge: pipeline_run → model (trains_on)
    await metadata_store.add_edge(
        source_id=pipeline_run.id,
        source_type="pipeline_run",
        target_id=artifact.id,
        target_type="model",
        edge_type="trains_on",
    )

    return artifact


async def load_tracked_model(
    model_id: uuid.UUID,
    metadata_store: MetadataStore,
    blob_store: BlobStore,
) -> tuple[Any, ModelArtifact]:
    artifact = await metadata_store.get_model(model_id)
    if artifact is None:
        raise ValueError(f"Model artifact {model_id} not found")

    model_bytes = await blob_store.async_download_bytes(artifact.storage_path)
    model = joblib.load(io.BytesIO(model_bytes))
    return model, artifact
