import logging
import subprocess
import uuid
from typing import Optional

from db.models import DatasetVersion, LineageEdge, ModelArtifact, PipelineRun
from store.metadata import MetadataStore

logger = logging.getLogger(__name__)


def _get_git_info() -> tuple[Optional[str], Optional[str]]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit, branch or None
    except Exception:
        return None, None


class PipelineTracker:
    def __init__(self, metadata_store: MetadataStore):
        self._meta = metadata_store

    async def start_run(
        self,
        name: str,
        parameters: Optional[dict] = None,
        created_by: str = "system",
    ) -> PipelineRun:
        git_commit, git_branch = _get_git_info()
        return await self._meta.create_pipeline_run(
            name=name,
            git_commit=git_commit,
            git_branch=git_branch,
            parameters=parameters or {},
            created_by=created_by,
        )

    async def finish_run(
        self,
        run_id: uuid.UUID,
        metrics: Optional[dict] = None,
        status: str = "success",
        error: Optional[str] = None,
    ) -> Optional[PipelineRun]:
        return await self._meta.finish_run(
            run_id=run_id,
            status=status,
            metrics=metrics,
            error_message=error,
        )

    async def link_input(self, run_id: uuid.UUID, dataset_version: DatasetVersion) -> LineageEdge:
        return await self._meta.add_edge(
            source_id=dataset_version.id,
            source_type="dataset",
            target_id=run_id,
            target_type="pipeline_run",
            edge_type="consumes",
        )

    async def link_output(self, run_id: uuid.UUID, dataset_version: DatasetVersion) -> LineageEdge:
        return await self._meta.add_edge(
            source_id=run_id,
            source_type="pipeline_run",
            target_id=dataset_version.id,
            target_type="dataset",
            edge_type="produces",
        )

    async def link_model(self, run_id: uuid.UUID, model: ModelArtifact) -> LineageEdge:
        return await self._meta.add_edge(
            source_id=run_id,
            source_type="pipeline_run",
            target_id=model.id,
            target_type="model",
            edge_type="trains_on",
        )

    async def get_run_lineage(self, run_id: uuid.UUID) -> dict:
        run = await self._meta.get_run(run_id)
        if run is None:
            raise ValueError(f"Pipeline run {run_id} not found")

        upstream = await self._meta.get_upstream(run_id, "pipeline_run")
        downstream = await self._meta.get_downstream(run_id, "pipeline_run")

        inputs: list[DatasetVersion] = []
        for edge in upstream:
            if edge.source_type == "dataset":
                dv = await self._meta.get_dataset_version(uuid.UUID(edge.source_id))
                if dv:
                    inputs.append(dv)

        outputs: list[DatasetVersion] = []
        models: list[ModelArtifact] = []
        for edge in downstream:
            if edge.target_type == "dataset":
                dv = await self._meta.get_dataset_version(uuid.UUID(edge.target_id))
                if dv:
                    outputs.append(dv)
            elif edge.target_type == "model":
                m = await self._meta.get_model(uuid.UUID(edge.target_id))
                if m:
                    models.append(m)

        return {"run": run, "inputs": inputs, "outputs": outputs, "models": models}


class tracked_run:
    def __init__(self, tracker: PipelineTracker, name: str, parameters: Optional[dict] = None):
        self._tracker = tracker
        self._name = name
        self._parameters = parameters or {}
        self._run: Optional[PipelineRun] = None

    async def __aenter__(self) -> PipelineRun:
        self._run = await self._tracker.start_run(self._name, self._parameters)
        return self._run

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._run is None:
            return False
        if exc_type is None:
            await self._tracker.finish_run(self._run.id, status="success")
        else:
            await self._tracker.finish_run(
                self._run.id,
                status="failed",
                error=str(exc_val),
            )
        return False
