import logging
import uuid
from typing import Optional

from store.metadata import MetadataStore
from db.models import PipelineRun, DatasetVersion, ModelArtifact
from core.lineage import LineageGraph

logger = logging.getLogger(__name__)


async def link_mlflow_run(
    mlflow_run_id: str,
    node_id: uuid.UUID,
    node_type: str,
    metadata_store: MetadataStore,
) -> None:
    """Store the MLflow run ID on a PipelineRun or ModelArtifact."""
    if node_type == "pipeline_run":
        run = await metadata_store.get_run(node_id)
        if run:
            run.git_commit = run.git_commit  # touch to keep session active
            # Store mlflow_run_id in parameters as a convention
            params = dict(run.parameters or {})
            params["mlflow_run_id"] = mlflow_run_id
            run.parameters = params
            await metadata_store._session.commit()
    elif node_type == "model":
        model = await metadata_store.get_model(node_id)
        if model:
            model.mlflow_run_id = mlflow_run_id
            await metadata_store._session.commit()


class LineageMLflowCallback:
    """Context manager that auto-links an active MLflow run to a lineage PipelineRun."""

    def __init__(self, pipeline_run: PipelineRun, metadata_store: MetadataStore):
        self._run = pipeline_run
        self._meta = metadata_store
        self._mlflow_run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            import mlflow
            active = mlflow.active_run()
            if active:
                import asyncio
                loop = asyncio.get_event_loop()
                loop.run_until_complete(
                    link_mlflow_run(
                        mlflow_run_id=active.info.run_id,
                        node_id=self._run.id,
                        node_type="pipeline_run",
                        metadata_store=self._meta,
                    )
                )
        except Exception as e:
            logger.warning("Failed to link MLflow run: %s", e)
        return False


async def get_mlflow_lineage(
    mlflow_run_id: str,
    metadata_store: MetadataStore,
) -> dict:
    """Given an MLflow run ID, return associated lineage nodes."""
    from sqlalchemy import select
    from db.models import PipelineRun

    result = await metadata_store._session.execute(
        select(PipelineRun)
    )
    all_runs = list(result.scalars().all())

    pipeline_run = None
    for run in all_runs:
        params = run.parameters or {}
        if params.get("mlflow_run_id") == mlflow_run_id:
            pipeline_run = run
            break

    if pipeline_run is None:
        return {
            "pipeline_run": None,
            "input_datasets": [],
            "output_model": None,
            "upstream_graph": {},
        }

    lg = LineageGraph(metadata_store)
    upstream = await lg.trace_upstream(str(pipeline_run.id), "pipeline_run")

    # Find input datasets from upstream edges
    upstream_edges = await metadata_store.get_upstream(pipeline_run.id, "pipeline_run")
    inputs: list[DatasetVersion] = []
    for edge in upstream_edges:
        if edge.source_type == "dataset":
            dv = await metadata_store.get_dataset_version(uuid.UUID(edge.source_id))
            if dv:
                inputs.append(dv)

    # Find output model from downstream edges
    downstream_edges = await metadata_store.get_downstream(pipeline_run.id, "pipeline_run")
    output_model: Optional[ModelArtifact] = None
    for edge in downstream_edges:
        if edge.target_type == "model":
            m = await metadata_store.get_model(uuid.UUID(edge.target_id))
            if m and m.mlflow_run_id == mlflow_run_id:
                output_model = m
                break

    return {
        "pipeline_run": pipeline_run,
        "input_datasets": inputs,
        "output_model": output_model,
        "upstream_graph": upstream["graph"],
    }


async def sync_mlflow_experiments(
    metadata_store: MetadataStore,
    mlflow_tracking_uri: str,
) -> int:
    """Scan MLflow experiments and link unlinked runs to lineage nodes."""
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()
    except ImportError:
        logger.error("mlflow not installed")
        return 0

    from sqlalchemy import select
    from db.models import PipelineRun

    result = await metadata_store._session.execute(select(PipelineRun))
    all_runs = list(result.scalars().all())

    # Build map of pipeline run name → run objects
    name_map: dict[str, list[PipelineRun]] = {}
    for run in all_runs:
        name_map.setdefault(run.name, []).append(run)

    linked = 0
    experiments = client.search_experiments()
    for exp in experiments:
        mlflow_runs = client.search_runs(experiment_ids=[exp.experiment_id])
        for mrun in mlflow_runs:
            run_name = mrun.data.tags.get("mlflow.runName", "")
            if run_name in name_map:
                for lineage_run in name_map[run_name]:
                    params = dict(lineage_run.parameters or {})
                    if "mlflow_run_id" not in params:
                        params["mlflow_run_id"] = mrun.info.run_id
                        lineage_run.parameters = params
                        linked += 1

    if linked:
        await metadata_store._session.commit()

    return linked
