import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DatasetVersion, PipelineRun, LineageEdge, ModelArtifact, PredictionLog


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MetadataStore:
    def __init__(self, session: AsyncSession):
        self._session = session

    # ── Dataset versions ──────────────────────────────────────────────────────

    async def create_dataset_version(
        self,
        name: str,
        content_hash: str,
        source_path: str,
        source_type: str,
        row_count: int,
        column_count: int,
        schema_json: dict,
        storage_path: str,
        created_by: str = "system",
        description: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> DatasetVersion:
        # Auto-increment version per name
        result = await self._session.execute(
            select(func.max(DatasetVersion.version)).where(DatasetVersion.name == name)
        )
        max_ver = result.scalar() or 0

        dv = DatasetVersion(
            name=name,
            version=max_ver + 1,
            content_hash=content_hash,
            source_path=source_path,
            source_type=source_type,
            row_count=row_count,
            column_count=column_count,
            schema_json=schema_json,
            storage_path=storage_path,
            created_by=created_by,
            description=description,
            tags=tags or [],
        )
        self._session.add(dv)
        await self._session.commit()
        await self._session.refresh(dv)
        return dv

    async def get_dataset_version(self, version_id: uuid.UUID) -> Optional[DatasetVersion]:
        result = await self._session.execute(
            select(DatasetVersion).where(DatasetVersion.id == version_id)
        )
        return result.scalar_one_or_none()

    async def get_latest_version(self, name: str) -> Optional[DatasetVersion]:
        result = await self._session.execute(
            select(DatasetVersion)
            .where(DatasetVersion.name == name)
            .order_by(DatasetVersion.version.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_all_versions(self, name: str) -> list[DatasetVersion]:
        result = await self._session.execute(
            select(DatasetVersion)
            .where(DatasetVersion.name == name)
            .order_by(DatasetVersion.version.asc())
        )
        return list(result.scalars().all())

    async def get_version_by_hash(self, content_hash: str) -> Optional[DatasetVersion]:
        result = await self._session.execute(
            select(DatasetVersion).where(DatasetVersion.content_hash == content_hash)
        )
        return result.scalar_one_or_none()

    async def list_datasets(self) -> list[str]:
        result = await self._session.execute(
            select(distinct(DatasetVersion.name)).order_by(DatasetVersion.name)
        )
        return list(result.scalars().all())

    # ── Pipeline runs ─────────────────────────────────────────────────────────

    async def create_pipeline_run(
        self,
        name: str,
        git_commit: Optional[str] = None,
        git_branch: Optional[str] = None,
        parameters: Optional[dict] = None,
        created_by: str = "system",
    ) -> PipelineRun:
        result = await self._session.execute(
            select(func.max(PipelineRun.run_number)).where(PipelineRun.name == name)
        )
        max_run = result.scalar() or 0

        run = PipelineRun(
            name=name,
            run_number=max_run + 1,
            status="running",
            git_commit=git_commit,
            git_branch=git_branch,
            parameters=parameters or {},
            metrics={},
            created_by=created_by,
        )
        self._session.add(run)
        await self._session.commit()
        await self._session.refresh(run)
        return run

    async def update_run_status(
        self,
        run_id: uuid.UUID,
        status: str,
        metrics: Optional[dict] = None,
        error_message: Optional[str] = None,
    ) -> Optional[PipelineRun]:
        run = await self.get_run(run_id)
        if run is None:
            return None
        run.status = status
        if metrics:
            run.metrics = metrics
        if error_message:
            run.error_message = error_message
        await self._session.commit()
        await self._session.refresh(run)
        return run

    async def finish_run(
        self,
        run_id: uuid.UUID,
        status: str,
        metrics: Optional[dict] = None,
        error_message: Optional[str] = None,
    ) -> Optional[PipelineRun]:
        run = await self.get_run(run_id)
        if run is None:
            return None
        now = _utcnow()
        run.status = status
        run.finished_at = now
        run.duration_seconds = (now - run.started_at).total_seconds()
        if metrics:
            run.metrics = metrics
        if error_message:
            run.error_message = error_message
        await self._session.commit()
        await self._session.refresh(run)
        return run

    async def get_run(self, run_id: uuid.UUID) -> Optional[PipelineRun]:
        result = await self._session.execute(
            select(PipelineRun).where(PipelineRun.id == run_id)
        )
        return result.scalar_one_or_none()

    async def list_runs(self, name: Optional[str] = None, limit: int = 50) -> list[PipelineRun]:
        q = select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(limit)
        if name:
            q = q.where(PipelineRun.name == name)
        result = await self._session.execute(q)
        return list(result.scalars().all())

    # ── Lineage edges ─────────────────────────────────────────────────────────

    async def add_edge(
        self,
        source_id: uuid.UUID,
        source_type: str,
        target_id: uuid.UUID,
        target_type: str,
        edge_type: str,
        metadata: Optional[dict] = None,
    ) -> LineageEdge:
        edge = LineageEdge(
            source_id=str(source_id),
            source_type=source_type,
            target_id=str(target_id),
            target_type=target_type,
            edge_type=edge_type,
            metadata_json=metadata,
        )
        self._session.add(edge)
        await self._session.commit()
        await self._session.refresh(edge)
        return edge

    async def get_upstream(self, node_id: uuid.UUID, node_type: str) -> list[LineageEdge]:
        result = await self._session.execute(
            select(LineageEdge).where(
                LineageEdge.target_id == str(node_id),
                LineageEdge.target_type == node_type,
            )
        )
        return list(result.scalars().all())

    async def get_downstream(self, node_id: uuid.UUID, node_type: str) -> list[LineageEdge]:
        result = await self._session.execute(
            select(LineageEdge).where(
                LineageEdge.source_id == str(node_id),
                LineageEdge.source_type == node_type,
            )
        )
        return list(result.scalars().all())

    async def get_all_edges(self) -> list[LineageEdge]:
        result = await self._session.execute(select(LineageEdge))
        return list(result.scalars().all())

    # ── Model artifacts ───────────────────────────────────────────────────────

    async def create_model_artifact(
        self,
        name: str,
        version: int,
        pipeline_run_id: uuid.UUID,
        storage_path: str,
        framework: str,
        mlflow_run_id: Optional[str] = None,
        mlflow_experiment_id: Optional[str] = None,
        metrics: Optional[dict] = None,
        parameters: Optional[dict] = None,
    ) -> ModelArtifact:
        artifact = ModelArtifact(
            name=name,
            version=version,
            pipeline_run_id=pipeline_run_id,
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment_id=mlflow_experiment_id,
            storage_path=storage_path,
            framework=framework,
            metrics=metrics or {},
            parameters=parameters or {},
        )
        self._session.add(artifact)
        await self._session.commit()
        await self._session.refresh(artifact)
        return artifact

    async def get_model(self, model_id: uuid.UUID) -> Optional[ModelArtifact]:
        result = await self._session.execute(
            select(ModelArtifact).where(ModelArtifact.id == model_id)
        )
        return result.scalar_one_or_none()

    async def get_model_by_name(self, name: str, version: Optional[int] = None) -> Optional[ModelArtifact]:
        q = select(ModelArtifact).where(ModelArtifact.name == name)
        if version is not None:
            q = q.where(ModelArtifact.version == version)
        else:
            q = q.order_by(ModelArtifact.version.desc()).limit(1)
        result = await self._session.execute(q)
        return result.scalar_one_or_none()

    # ── Predictions ───────────────────────────────────────────────────────────

    async def log_prediction(
        self,
        model_artifact_id: uuid.UUID,
        entity_id: str,
        input_features: dict,
        prediction: float,
        dataset_version_id: Optional[uuid.UUID] = None,
    ) -> PredictionLog:
        log = PredictionLog(
            model_artifact_id=model_artifact_id,
            entity_id=entity_id,
            input_features=input_features,
            prediction=prediction,
            dataset_version_id=dataset_version_id,
        )
        self._session.add(log)
        await self._session.commit()
        await self._session.refresh(log)
        return log

    async def get_prediction(self, prediction_id: uuid.UUID) -> Optional[PredictionLog]:
        result = await self._session.execute(
            select(PredictionLog).where(PredictionLog.id == prediction_id)
        )
        return result.scalar_one_or_none()

    async def list_predictions(self, model_artifact_id: uuid.UUID, limit: int = 100) -> list[PredictionLog]:
        result = await self._session.execute(
            select(PredictionLog)
            .where(PredictionLog.model_artifact_id == model_artifact_id)
            .order_by(PredictionLog.predicted_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
