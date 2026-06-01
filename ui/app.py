import asyncio
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from db.base import AsyncSessionLocal, create_all_tables
from store.metadata import MetadataStore
from store.blob import BlobStore
from core.dataset import DatasetTracker
from core.lineage import LineageGraph


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


st.set_page_config(
    page_title="Pipeline Lineage Tracker",
    page_icon="🔗",
    layout="wide",
)

st.sidebar.title("🔗 Lineage Tracker")
page = st.sidebar.radio(
    "Navigate",
    ["Dataset Versions", "Pipeline Runs", "Lineage DAG", "Impact Analysis"],
)

# ── Page 1: Dataset Versions ─────────────────────────────────────────────────

if page == "Dataset Versions":
    st.title("Dataset Versions")

    async def get_datasets():
        async with AsyncSessionLocal() as session:
            meta = MetadataStore(session)
            return await meta.list_datasets()

    datasets = run_async(get_datasets())

    if not datasets:
        st.info("No datasets tracked yet. Run the demo to ingest data.")
        st.stop()

    selected = st.selectbox("Select dataset", datasets)

    async def get_versions(name):
        async with AsyncSessionLocal() as session:
            meta = MetadataStore(session)
            return await meta.get_all_versions(name)

    versions = run_async(get_versions(selected))

    if versions:
        rows = []
        for v in versions:
            rows.append({
                "Version": v.version,
                "Hash (8 chars)": v.content_hash[:8],
                "Rows": v.row_count,
                "Columns": v.column_count,
                "Created At": v.created_at.strftime("%Y-%m-%d %H:%M"),
                "Description": v.description or "",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # Schema expander
        for v in versions:
            with st.expander(f"Schema — v{v.version}"):
                schema = v.schema_json or {}
                if schema:
                    st.dataframe(
                        pd.DataFrame([{"Column": k, "Type": val} for k, val in schema.items()]),
                        use_container_width=True,
                    )
                else:
                    st.write("No schema info available.")

        # Diff
        st.subheader("Version Diff")
        col1, col2 = st.columns(2)
        version_numbers = [v.version for v in versions]
        va = col1.selectbox("Version A", version_numbers, index=0)
        vb = col2.selectbox("Version B", version_numbers, index=min(1, len(version_numbers) - 1))

        if st.button("Compare Versions") and va != vb:
            async def do_diff(name, a, b):
                async with AsyncSessionLocal() as session:
                    meta = MetadataStore(session)
                    blob = BlobStore()
                    tracker = DatasetTracker(meta, blob)
                    return await tracker.diff(name, a, b)

            try:
                diff = run_async(do_diff(selected, va, vb))
                st.write(f"**Row count delta:** {diff['row_count_delta']:+d}")
                st.write(f"**Content identical:** {diff['content_identical']}")
                sc = diff["schema_changes"]
                if sc["added"]:
                    st.success(f"Added columns: {', '.join(sc['added'].keys())}")
                if sc["removed"]:
                    st.error(f"Removed columns: {', '.join(sc['removed'].keys())}")
                if sc["changed"]:
                    st.warning(f"Changed dtypes: {', '.join(sc['changed'].keys())}")
                if not any([sc["added"], sc["removed"], sc["changed"]]):
                    st.info("No schema changes between these versions.")
            except ValueError as e:
                st.error(str(e))

# ── Page 2: Pipeline Runs ─────────────────────────────────────────────────────

elif page == "Pipeline Runs":
    st.title("Pipeline Runs")

    async def get_runs(name_filter=None, limit=50):
        async with AsyncSessionLocal() as session:
            meta = MetadataStore(session)
            return await meta.list_runs(name=name_filter or None, limit=limit)

    col1, col2 = st.columns([3, 1])
    name_filter = col1.text_input("Filter by pipeline name", "")
    limit = col2.number_input("Max rows", min_value=5, max_value=500, value=50)

    runs = run_async(get_runs(name_filter, int(limit)))

    if not runs:
        st.info("No pipeline runs found.")
    else:
        status_badge = {"success": "🟢", "failed": "🔴", "running": "🟡"}
        rows = []
        for r in runs:
            rows.append({
                "Run #": r.run_number,
                "Name": r.name,
                "Status": f"{status_badge.get(r.status, '⚪')} {r.status}",
                "Git Commit": r.git_commit[:8] if r.git_commit else "-",
                "Duration": f"{r.duration_seconds:.1f}s" if r.duration_seconds else "-",
                "Started": r.started_at.strftime("%Y-%m-%d %H:%M"),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        for r in runs:
            with st.expander(f"Run #{r.run_number} — {r.name}"):
                col1, col2 = st.columns(2)
                col1.json(r.parameters or {})
                col2.json(r.metrics or {})
                if r.error_message:
                    st.error(r.error_message)

# ── Page 3: Lineage DAG ───────────────────────────────────────────────────────

elif page == "Lineage DAG":
    st.title("Lineage DAG")

    async def get_graph_html(filter_types=None):
        async with AsyncSessionLocal() as session:
            meta = MetadataStore(session)
            lg = LineageGraph(meta)
            G = await lg.build_graph()

            if filter_types:
                nodes_to_keep = [
                    n for n, d in G.nodes(data=True)
                    if d.get("node_type") in filter_types
                ]
                G = G.subgraph(nodes_to_keep).copy()

            stats = {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
            }
            html = await lg.to_pyvis_html(G)
            return html, stats

    with st.sidebar:
        st.subheader("Graph Filters")
        show_types = st.multiselect(
            "Show node types",
            ["dataset", "pipeline_run", "model", "prediction"],
            default=["dataset", "pipeline_run", "model", "prediction"],
        )

    html, stats = run_async(get_graph_html(show_types if show_types else None))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Nodes", stats["nodes"])
    col2.metric("Total Edges", stats["edges"])
    col3.metric("Node Types", len(show_types))
    col4.metric("Graph", "Live")

    components.html(html, height=620, scrolling=False)

    with st.sidebar:
        st.subheader("Trace Upstream")
        trace_id = st.text_input("Node ID (UUID)")
        trace_type = st.selectbox("Node type", ["dataset", "pipeline_run", "model", "prediction"])
        if st.button("Trace") and trace_id:
            async def do_trace(nid, ntype):
                async with AsyncSessionLocal() as session:
                    meta = MetadataStore(session)
                    lg = LineageGraph(meta)
                    return await lg.trace_upstream(nid, ntype)

            result = run_async(do_trace(trace_id, trace_type))
            st.write(f"**Depth:** {result['depth']}")
            st.write("**Path:**")
            for step in result["path"]:
                st.code(step)

# ── Page 4: Impact Analysis ───────────────────────────────────────────────────

elif page == "Impact Analysis":
    st.title("Impact Analysis")
    st.write("Understand what breaks downstream if a dataset changes.")

    async def get_dataset_names():
        async with AsyncSessionLocal() as session:
            meta = MetadataStore(session)
            return await meta.list_datasets()

    dataset_names = run_async(get_dataset_names())

    if not dataset_names:
        st.info("No datasets tracked yet.")
        st.stop()

    col1, col2 = st.columns([3, 1])
    ds_name = col1.selectbox("Dataset", dataset_names)
    ds_version = col2.number_input("Version (0 = latest)", min_value=0, value=0)

    if st.button("Run Impact Analysis"):
        async def do_impact(name, version):
            async with AsyncSessionLocal() as session:
                meta = MetadataStore(session)
                lg = LineageGraph(meta)
                result = await lg.impact_analysis(name, version if version > 0 else None)

                # Also get subgraph HTML
                latest_dv = await meta.get_latest_version(name)
                subgraph_html = ""
                if latest_dv:
                    downstream = await lg.trace_downstream(str(latest_dv.id), "dataset")
                    import networkx as nx
                    G = await lg.build_graph()
                    node_keys = {n["key"] for n in downstream["graph"]["nodes"]}
                    subgraph = G.subgraph(node_keys).copy()
                    subgraph_html = await lg.to_pyvis_html(subgraph) if subgraph.number_of_nodes() > 0 else ""

                return result, subgraph_html

        result, subgraph_html = run_async(do_impact(ds_name, int(ds_version)))

        risk = result["risk_level"]
        risk_color = {"low": "green", "medium": "orange", "high": "red"}.get(risk, "gray")

        st.markdown(f"### Risk Level: :{risk_color}[{risk.upper()}]")

        col1, col2, col3 = st.columns(3)
        col1.metric("Affected Pipeline Runs", len(result["affected_pipeline_runs"]))
        col2.metric("Affected Models", len(result["affected_models"]))
        col3.metric("Affected Predictions", result["affected_predictions_count"])

        if result["affected_models"]:
            st.subheader("Models at Risk")
            for m in result["affected_models"]:
                st.write(f"• **{m.name}** v{m.version} ({m.framework})")

        if subgraph_html:
            st.subheader("Downstream Lineage")
            components.html(subgraph_html, height=400, scrolling=False)
