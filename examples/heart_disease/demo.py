"""
End-to-end demo: heart disease dataset -> versioning -> lineage -> impact analysis.
Run: python examples/heart_disease/demo.py
"""
import asyncio
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.dataset import DatasetTracker
from core.lineage import LineageGraph
from core.pipeline import PipelineTracker
from db.base import AsyncSessionLocal, create_all_tables
from store.blob import BlobStore
from store.metadata import MetadataStore

DATA_DIR = Path(__file__).parent / "data"
DATA_PATH = DATA_DIR / "heart.csv"
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)
CLEVELAND_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

SEP = "-" * 60


def _download_heart_csv() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print("  Downloading heart disease dataset from UCI...")
        urllib.request.urlretrieve(UCI_URL, DATA_PATH)
        df = pd.read_csv(DATA_PATH, names=CLEVELAND_COLS, na_values="?")
        df.dropna(inplace=True)
        df["target"] = (df["target"] > 0).astype(int)
        df.to_csv(DATA_PATH, index=False)
    return DATA_PATH


async def run_demo():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        blob = BlobStore()
        dataset_tracker = DatasetTracker(meta, blob)
        pipeline_tracker = PipelineTracker(meta)
        lg = LineageGraph(meta)

        # Step 1: Track raw dataset
        print(f"\n[1/8] Track raw dataset\n{SEP}")
        csv_path = _download_heart_csv()
        raw_version = await dataset_tracker.track_file(
            "heart_raw",
            str(csv_path),
            source_type="csv",
            description="UCI Cleveland Heart Disease dataset (303 patients)",
            tags=["raw", "medical", "heart"],
        )
        print(f"  OK  heart_raw tracked as version {raw_version.version}, "
              f"hash {raw_version.content_hash[:8]}, {raw_version.row_count} rows")

        # Step 2: Feature engineering pipeline run
        print(f"\n[2/8] Feature engineering pipeline run\n{SEP}")
        df_raw = pd.read_csv(csv_path)

        fe_run = await pipeline_tracker.start_run("feature_engineering", {"n_features": 3})
        await pipeline_tracker.link_input(fe_run.id, raw_version)

        df_features = pd.DataFrame()
        df_features["patient_id"] = range(len(df_raw))
        age = df_raw["age"]
        df_features["age_normalized"] = (age - age.min()) / (age.max() - age.min())
        df_features["cholesterol_risk"] = (df_raw["chol"] > 240).astype(float)
        df_features["composite_risk"] = (
            0.4 * df_features["age_normalized"]
            + 0.3 * df_features["cholesterol_risk"]
            + 0.3 * df_raw["exang"]
        )
        df_features["target"] = df_raw["target"].values

        features_version = await dataset_tracker.track_dataframe(
            "heart_features",
            df_features,
            source_type="dataframe",
            description="Engineered features from heart_raw",
            tags=["features", "heart"],
        )
        await pipeline_tracker.link_output(fe_run.id, features_version)
        await pipeline_tracker.finish_run(
            fe_run.id,
            metrics={"n_features": 3, "n_rows": len(df_features)},
        )
        print(f"  OK  Feature engineering run recorded, 3 features computed, "
              f"heart_features v{features_version.version}")

        # Step 3: Train a model
        print(f"\n[3/8] Train a model\n{SEP}")
        feature_cols = ["age_normalized", "cholesterol_risk", "composite_risk"]
        X = df_features[feature_cols].values
        y = df_features["target"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        }

        training_run = await pipeline_tracker.start_run(
            "model_training",
            parameters={"n_estimators": 100, "max_depth": 3},
        )
        await pipeline_tracker.link_input(training_run.id, features_version)

        from integrations.sklearn import track_sklearn_model

        model_artifact = await track_sklearn_model(
            model={"scaler": scaler, "clf": clf},
            name="heart_disease_model",
            version=1,
            pipeline_run=training_run,
            metadata_store=meta,
            blob_store=blob,
            metrics=metrics,
            parameters={"n_estimators": 100, "max_depth": 3},
        )

        await pipeline_tracker.link_model(training_run.id, model_artifact)
        await pipeline_tracker.finish_run(training_run.id, metrics=metrics)
        print(f"  OK  Model trained -- accuracy={metrics['accuracy']:.3f}, "
              f"f1={metrics['f1']:.3f}, roc_auc={metrics['roc_auc']:.3f}")
        print(f"  OK  Artifact saved: {model_artifact.storage_path}")

        # Step 4: Log predictions
        print(f"\n[4/8] Log predictions\n{SEP}")
        sample = X_test_scaled[:5]
        preds = clf.predict_proba(sample)[:, 1]
        first_pred_id = None

        for i, (feat_row, prob) in enumerate(zip(X_test[:5], preds)):
            features_dict = {
                "age_normalized": float(feat_row[0]),
                "cholesterol_risk": float(feat_row[1]),
                "composite_risk": float(feat_row[2]),
            }
            pred_log = await meta.log_prediction(
                model_artifact_id=model_artifact.id,
                entity_id=str(i),
                input_features=features_dict,
                prediction=float(prob),
                dataset_version_id=features_version.id,
            )
            await meta.add_edge(
                source_id=model_artifact.id,
                source_type="model",
                target_id=pred_log.id,
                target_type="prediction",
                edge_type="predicts_with",
            )
            if first_pred_id is None:
                first_pred_id = pred_log.id
                print(f"  First prediction ID: {pred_log.id}")

        print(f"  OK  5 predictions logged")

        # Step 5: Dataset change — new version
        print(f"\n[5/8] Simulate dataset change (new version)\n{SEP}")
        df_v2 = df_raw.copy()
        df_v2["bmi_proxy"] = df_v2["age"] * 0.3 + df_v2["chol"] * 0.01
        df_v2.iloc[:5, df_v2.columns.get_loc("age")] += 1
        df_v2.to_csv(DATA_PATH, index=False)

        raw_v2 = await dataset_tracker.track_file(
            "heart_raw",
            str(csv_path),
            source_type="csv",
            description="heart_raw with bmi_proxy column added",
            tags=["raw", "medical", "heart", "v2"],
        )
        diff = await dataset_tracker.diff("heart_raw", 1, raw_v2.version)
        print(f"  OK  Dataset v{raw_v2.version} tracked")
        print(f"      Hash changed: {diff['hash_changed']}")
        sc = diff["schema_changes"]
        if sc["added"]:
            print(f"      Added columns: {list(sc['added'].keys())}")
        print(f"      Row count delta: {diff['row_count_delta']:+d}")

        # Step 6: Impact analysis
        print(f"\n[6/8] Impact analysis\n{SEP}")
        impact = await lg.impact_analysis("heart_raw", version=1)
        print(f"  Affected pipeline runs: {len(impact['affected_pipeline_runs'])}")
        print(f"  Affected models:        {len(impact['affected_models'])}")
        print(f"  Affected predictions:   {impact['affected_predictions_count']}")
        print(f"  Risk level:             {impact['risk_level'].upper()}")
        print("  OK  Impact analysis complete")

        # Step 7: Trace a prediction
        print(f"\n[7/8] Trace prediction lineage\n{SEP}")
        if first_pred_id:
            trace = await lg.trace_upstream(str(first_pred_id), "prediction")
            print(f"  Ancestry chain ({trace['depth']} hops):")
            for node in trace["path"]:
                print(f"    -> {node}")
            print(f"  OK  Full lineage traced: {trace['depth']} hops from raw data to prediction")

        # Step 8: Export DAG
        print(f"\n[8/8] Export interactive DAG\n{SEP}")
        G = await lg.build_graph()
        html = await lg.to_pyvis_html(G)
        out_path = Path(__file__).parent / "lineage.html"
        out_path.write_text(html, encoding="utf-8")
        print(f"  OK  Interactive DAG saved to {out_path}")
        print(f"      ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


if __name__ == "__main__":
    asyncio.run(create_all_tables())
    asyncio.run(run_demo())
