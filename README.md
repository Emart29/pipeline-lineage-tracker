# Pipeline Lineage Tracker

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql&logoColor=white)
![MinIO](https://img.shields.io/badge/MinIO-Object_Storage-red?logo=minio&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Trace any prediction back to the exact raw data that produced it. SHA-256 dataset versioning, full pipeline lineage DAG, impact analysis, and an interactive graph explorer — all in one repo.

## Tech Stack

| Layer | Tool |
| --- | --- |
| Metadata store | PostgreSQL + SQLAlchemy (async) |
| Snapshot store | MinIO (parquet + raw files) |
| Graph engine | NetworkX |
| DAG visualization | pyvis (interactive HTML) |
| CLI | Click + Rich |
| UI | Streamlit |
| Pandas integration | @track_dataframe decorator |
| sklearn integration | TrackedPipeline wrapper |
| MLflow integration | LineageMLflowCallback |

---

## Architecture

```text
Raw Data (CSV, DB, Kafka)
         │
         ▼
  DatasetTracker
  ├── SHA-256 hash (dedup identical content)
  ├── Upload snapshot → MinIO
  └── Register DatasetVersion (PostgreSQL)
         │
         ▼
  PipelineTracker ──────────────────────────────────────────┐
  ├── start_run() → PipelineRun (git commit + branch)       │
  ├── link_input(run, dataset_version)                       │
  ├── link_output(run, dataset_version)                      │
  └── finish_run(metrics)                                    │
         │                                                   │
         ▼                                                   ▼
  Feature DatasetVersion              ModelArtifact (serialized to MinIO)
         │                                   │
         └──────────────┬────────────────────┘
                        ▼
                  LineageGraph (NetworkX DAG)
                  ├── trace_upstream(prediction) → raw CSV → run → features → model → prediction
                  ├── impact_analysis(dataset) → risk level + affected models
                  └── to_pyvis_html() → interactive graph
                        │
                        ▼
                  Streamlit UI (port 8502)
                  ├── Page 1: Dataset Versions + Diff
                  ├── Page 2: Pipeline Runs
                  ├── Page 3: Interactive Lineage DAG
                  └── Page 4: Impact Analysis
```

## Services

| Service | URL | Description |
| --- | --- | --- |
| Streamlit UI | <http://localhost:8502> | Dataset versions, pipeline runs, lineage DAG, impact analysis |

---

## Quick Start — Docker

**Prerequisites:** Docker Desktop running + shared infrastructure stack.

```bash
# 1. Start shared infrastructure (PostgreSQL + MinIO + Redis + Kafka)
git clone https://github.com/Emart29/ml-platform-infra
cd ml-platform-infra && docker compose up -d && cd ..

# 2. Clone and start the lineage tracker
git clone https://github.com/Emart29/pipeline-lineage-tracker
cd pipeline-lineage-tracker
docker compose up
```

Docker will automatically:

1. Create the database schema (5 tables: dataset_versions, pipeline_runs, lineage_edges, model_artifacts, prediction_logs)
2. Download the Heart Disease dataset (303 patients) and run the full demo
3. Track 2 dataset versions, 2 pipeline runs, 1 model artifact, 5 predictions
4. Build the lineage DAG and export it to HTML
5. Start the Streamlit UI on port 8502

Open **<http://localhost:8502>** — the lineage graph is live.

```bash
# Stop containers
docker compose down

# Stop and delete all data
docker compose down -v
```

---

## Local Setup

Requires Python 3.10+.

```bash
# 1. Start shared infrastructure
git clone https://github.com/Emart29/ml-platform-infra
cd ml-platform-infra && docker compose up -d && cd ..

# 2. Install
git clone https://github.com/Emart29/pipeline-lineage-tracker
cd pipeline-lineage-tracker
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
pip install -e .

# 3. Configure
copy .env.example .env

# 4. Initialize + run demo
python -c "import asyncio; from db.base import create_all_tables; asyncio.run(create_all_tables())"
python examples/heart_disease/demo.py

# 5. Start UI
streamlit run ui/app.py --server.port 8502
```

---

## CLI

```bash
# List all tracked datasets
lineage datasets

# Show all versions of a dataset
lineage versions heart_raw

# Diff two versions
lineage diff heart_raw 1 2

# List pipeline runs
lineage runs
lineage runs --name feature_engineering --limit 10

# Trace full upstream lineage of any node
lineage trace <uuid> --type prediction
lineage trace <uuid> --type model

# Impact analysis — what breaks if this dataset changes?
lineage impact heart_raw
lineage impact heart_raw --version 1

# Export interactive lineage DAG to HTML
lineage graph --output lineage.html --open

# Summary statistics
lineage stats
```

Example `lineage stats` output:

```text
╭──────────────────────┬───────╮
│ Metric               │ Value │
├──────────────────────┼───────┤
│ Total nodes          │    12 │
│ Total edges          │    11 │
│ Dataset versions     │     3 │
│ Pipeline runs        │     2 │
│ Models               │     1 │
│ Predictions          │     5 │
│ Avg lineage depth    │  4.00 │
╰──────────────────────┴───────╯
```

---

## Python API

### Track a file

```python
from db.base import AsyncSessionLocal, create_all_tables
from store.metadata import MetadataStore
from store.blob import BlobStore
from core.dataset import DatasetTracker

async with AsyncSessionLocal() as session:
    meta = MetadataStore(session)
    blob = BlobStore()
    tracker = DatasetTracker(meta, blob)

    version = await tracker.track_file(
        "heart_raw",
        "data/heart.csv",
        source_type="csv",
        description="Raw UCI Cleveland dataset",
        tags=["raw", "medical"],
    )
    print(f"Tracked as v{version.version}, hash: {version.content_hash[:8]}")
    # Tracked as v1, hash: a3f8c21d
```

### Track a pipeline run

```python
from core.pipeline import PipelineTracker, tracked_run

async with AsyncSessionLocal() as session:
    meta = MetadataStore(session)
    pipeline_tracker = PipelineTracker(meta)

    async with tracked_run(pipeline_tracker, "feature_engineering", {"n_features": 3}) as run:
        # do work — auto-records git commit, start/end time, status
        await pipeline_tracker.link_input(run.id, raw_version)
        # ... engineer features ...
        await pipeline_tracker.link_output(run.id, features_version)
    # run auto-finished as "success" on context exit
```

### Trace lineage

```python
from core.lineage import LineageGraph

async with AsyncSessionLocal() as session:
    meta = MetadataStore(session)
    lg = LineageGraph(meta)

    # Trace a prediction all the way back to raw data
    trace = await lg.trace_upstream(prediction_id, "prediction")
    print(f"Depth: {trace['depth']} hops")
    for step in trace["path"]:
        print(f"  → {step}")
    # → dataset:a1b2c3...  (heart_raw v1)
    # → pipeline_run:d4e5f6...  (feature_engineering #1)
    # → dataset:g7h8i9...  (heart_features v1)
    # → pipeline_run:j1k2l3...  (model_training #1)
    # → model:m4n5o6...  (heart_disease_model v1)
    # → prediction:p7q8r9...
```

---

## Integrations

### Pandas

```python
from integrations.pandas import track_dataframe, track_read_csv, compare_dataframes

# Auto-track the returned DataFrame
@track_dataframe("heart_features", tracker=dataset_tracker)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... transform ...
    return features_df

# Track a CSV read
df, version = await track_read_csv("data/heart.csv", "heart_raw", dataset_tracker)
print(f"Loaded v{version.version}, hash {version.content_hash[:8]}")

# Compare two DataFrames for drift
diff = compare_dataframes(df_v1, df_v2)
print(diff["numeric_column_stats"])  # mean/std shift per column
print(diff["new_nulls"])             # columns with increased null rate
```

### sklearn

```python
from integrations.sklearn import TrackedPipeline

pipeline = TrackedPipeline(
    steps=[("scaler", StandardScaler()), ("clf", GradientBoostingClassifier())],
    pipeline_tracker=pipeline_tracker,
    dataset_tracker=dataset_tracker,
    metadata_store=meta,
    blob_store=blob,
    pipeline_name="model_training",
    model_name="heart_disease_model",
    parameters={"n_estimators": 100},
)
pipeline.fit(X_train, y_train)
# Automatically: starts run, trains, serializes to MinIO, records ModelArtifact, links edges
print(pipeline.lineage_model.storage_path)  # models/heart_disease_model/v1/heart_disease_model.joblib
```

### MLflow

```python
from integrations.mlflow import LineageMLflowCallback
import mlflow

with LineageMLflowCallback(pipeline_run=run, metadata_store=meta):
    with mlflow.start_run() as mlflow_run:
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        mlflow.log_metrics(metrics)
# On exit: LineageMLflowCallback links the MLflow run ID to the lineage PipelineRun
```

---

## Streamlit UI

| Page | URL fragment | Description |
| --- | --- | --- |
| Dataset Versions | Page 1 | Browse versions, compare schemas, download snapshots |
| Pipeline Runs | Page 2 | Filter runs by name/status, inspect parameters and metrics |
| Lineage DAG | Page 3 | Interactive pyvis graph, trace upstream, filter by node type |
| Impact Analysis | Page 4 | Risk level badge, affected models list, downstream subgraph |

---

## Project Layout

```text
pipeline-lineage-tracker/
├── core/
│   ├── dataset.py      # DatasetTracker — track_file, track_dataframe, diff, load
│   ├── pipeline.py     # PipelineTracker + tracked_run context manager
│   ├── lineage.py      # LineageGraph — NetworkX DAG, trace, impact, pyvis export
│   └── snapshot.py     # SHA-256 hashing + MinIO snapshot helpers
├── store/
│   ├── metadata.py     # All PostgreSQL CRUD (async SQLAlchemy)
│   └── blob.py         # MinIO wrapper (upload/download/exists)
├── integrations/
│   ├── pandas.py       # @track_dataframe decorator, track_read_csv, compare_dataframes
│   ├── sklearn.py      # TrackedPipeline, track_sklearn_model, load_tracked_model
│   └── mlflow.py       # LineageMLflowCallback, link_mlflow_run, get_mlflow_lineage
├── cli/
│   └── main.py         # 8-command Click CLI (datasets, versions, diff, runs, trace, impact, graph, stats)
├── ui/
│   └── app.py          # 4-page Streamlit app
├── db/
│   ├── models.py       # 5 SQLAlchemy ORM models
│   └── base.py         # Async engine (NullPool) + session factory
├── examples/
│   └── heart_disease/
│       └── demo.py     # 8-step end-to-end demo script
├── scripts/
│   └── docker_init.py  # One-shot Docker setup
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

---

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `POSTGRES_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/ml_platform` | PostgreSQL connection string |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO endpoint (host:port) |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET` | `lineage-snapshots` | Bucket for dataset and model snapshots |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Plugs into Project 1

This project is designed to complement [ml-feature-store](https://github.com/Emart29/ml-feature-store).

Feature datasets produced by the feature store are tracked as `DatasetVersion` nodes in the lineage graph. This means you can:

1. Trace any online prediction (from the feature store's Redis serving layer) back through the feature ingestion pipeline to the original raw data file — with the exact git commit hash of the transform function that produced each feature.
2. Run `lineage impact heart_raw --version 1` to see which feature store ingestion runs, model training runs, and live predictions would be affected by a dataset change — before you make it.
3. Use the Streamlit UI to visually explore the full MLOps graph: raw data → features → model → predictions, all linked by SHA-256 content hashes.
