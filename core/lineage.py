import uuid
from typing import Optional

import networkx as nx
from pyvis.network import Network

from db.models import DatasetVersion, LineageEdge, ModelArtifact, PipelineRun
from store.metadata import MetadataStore

_NODE_COLORS = {
    "dataset": "#4A90D9",
    "pipeline_run": "#7ED321",
    "model": "#F5A623",
    "prediction": "#D0021B",
}

_NODE_SHAPES = {
    "dataset": "dot",
    "pipeline_run": "box",
    "model": "diamond",
    "prediction": "triangle",
}


def _node_key(node_type: str, node_id: str) -> str:
    return f"{node_type}:{node_id}"


class LineageGraph:
    def __init__(self, metadata_store: MetadataStore):
        self._meta = metadata_store

    async def build_graph(self) -> nx.DiGraph:
        edges = await self._meta.get_all_edges()
        G = nx.DiGraph()

        for edge in edges:
            src_key = _node_key(edge.source_type, edge.source_id)
            tgt_key = _node_key(edge.target_type, edge.target_id)

            G.add_node(src_key, node_type=edge.source_type, node_id=edge.source_id)
            G.add_node(tgt_key, node_type=edge.target_type, node_id=edge.target_id)
            G.add_edge(src_key, tgt_key, edge_type=edge.edge_type)

        # Enrich node labels
        await self._enrich_labels(G)
        return G

    async def _enrich_labels(self, G: nx.DiGraph) -> None:
        for node_key, attrs in G.nodes(data=True):
            node_type = attrs.get("node_type", "")
            node_id = attrs.get("node_id", "")
            label = node_key  # fallback
            try:
                uid = uuid.UUID(node_id)
                if node_type == "dataset":
                    dv = await self._meta.get_dataset_version(uid)
                    if dv:
                        label = f"{dv.name}\nv{dv.version}"
                elif node_type == "pipeline_run":
                    run = await self._meta.get_run(uid)
                    if run:
                        label = f"{run.name}\n#{run.run_number}"
                elif node_type == "model":
                    m = await self._meta.get_model(uid)
                    if m:
                        label = f"{m.name}\nv{m.version}"
                elif node_type == "prediction":
                    label = f"prediction\n{node_id[:8]}"
            except Exception:
                pass
            G.nodes[node_key]["label"] = label

    async def trace_upstream(self, node_id: str, node_type: str) -> dict:
        G = await self.build_graph()
        start = _node_key(node_type, node_id)
        if start not in G:
            return {"path": [], "graph": {"nodes": [], "edges": []}, "depth": 0}

        ancestors = nx.ancestors(G, start)
        ancestors.add(start)
        subgraph = G.subgraph(ancestors)

        try:
            paths = list(nx.all_simple_paths(G, source=min(
                (n for n in ancestors if G.in_degree(n) == 0), key=len
            ), target=start))
            longest = max(paths, key=len) if paths else [start]
        except Exception:
            longest = [start]

        return {
            "path": longest,
            "graph": self._graph_to_dict(subgraph),
            "depth": len(longest) - 1,
        }

    async def trace_downstream(self, node_id: str, node_type: str) -> dict:
        G = await self.build_graph()
        start = _node_key(node_type, node_id)
        if start not in G:
            return {"path": [], "graph": {"nodes": [], "edges": []}, "depth": 0}

        descendants = nx.descendants(G, start)
        descendants.add(start)
        subgraph = G.subgraph(descendants)

        try:
            leaves = [n for n in descendants if G.out_degree(n) == 0]
            paths = [p for leaf in leaves for p in nx.all_simple_paths(G, source=start, target=leaf)]
            longest = max(paths, key=len) if paths else [start]
        except Exception:
            longest = [start]

        return {
            "path": longest,
            "graph": self._graph_to_dict(subgraph),
            "depth": len(longest) - 1,
        }

    async def impact_analysis(self, dataset_name: str, version: Optional[int] = None) -> dict:
        dv = await self._meta.get_latest_version(dataset_name) if version is None else None
        if version is not None:
            versions = await self._meta.get_all_versions(dataset_name)
            dv = next((v for v in versions if v.version == version), None)

        if dv is None:
            return {
                "affected_pipeline_runs": [],
                "affected_models": [],
                "affected_predictions_count": 0,
                "risk_level": "low",
            }

        downstream = await self.trace_downstream(str(dv.id), "dataset")
        graph_dict = downstream["graph"]

        pipeline_runs: list[PipelineRun] = []
        models: list[ModelArtifact] = []
        prediction_count = 0

        for node in graph_dict["nodes"]:
            ntype = node.get("node_type")
            nid = node.get("node_id")
            if not nid:
                continue
            try:
                uid = uuid.UUID(nid)
                if ntype == "pipeline_run":
                    run = await self._meta.get_run(uid)
                    if run:
                        pipeline_runs.append(run)
                elif ntype == "model":
                    m = await self._meta.get_model(uid)
                    if m:
                        models.append(m)
                elif ntype == "prediction":
                    prediction_count += 1
            except Exception:
                pass

        model_count = len(models)
        if model_count == 0:
            risk_level = "low"
        elif model_count <= 2:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "affected_pipeline_runs": pipeline_runs,
            "affected_models": models,
            "affected_predictions_count": prediction_count,
            "risk_level": risk_level,
        }

    async def to_pyvis_html(self, graph: nx.DiGraph) -> str:
        net = Network(height="600px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")
        net.force_atlas_2based()

        for node_key, attrs in graph.nodes(data=True):
            node_type = attrs.get("node_type", "dataset")
            label = attrs.get("label", node_key)
            net.add_node(
                node_key,
                label=label,
                color=_NODE_COLORS.get(node_type, "#888888"),
                shape=_NODE_SHAPES.get(node_type, "dot"),
                size=25,
                font={"size": 12, "color": "white"},
            )

        for src, tgt, attrs in graph.edges(data=True):
            net.add_edge(src, tgt, label=attrs.get("edge_type", ""), color="#aaaaaa")

        net.set_options("""{
          "physics": {
            "enabled": true,
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "springLength": 120
            },
            "solver": "forceAtlas2Based"
          }
        }""")

        return net.generate_html()

    async def get_stats(self) -> dict:
        G = await self.build_graph()
        node_types = {t: 0 for t in ("dataset", "pipeline_run", "model", "prediction")}
        for _, attrs in G.nodes(data=True):
            t = attrs.get("node_type", "")
            if t in node_types:
                node_types[t] += 1

        depths = []
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        for root in roots:
            descendants = nx.descendants(G, root)
            for d in descendants:
                try:
                    length = nx.shortest_path_length(G, root, d)
                    depths.append(length)
                except Exception:
                    pass

        avg_depth = sum(depths) / len(depths) if depths else 0.0

        return {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "dataset_versions": node_types["dataset"],
            "pipeline_runs": node_types["pipeline_run"],
            "models": node_types["model"],
            "predictions": node_types["prediction"],
            "avg_lineage_depth": round(avg_depth, 2),
        }

    @staticmethod
    def _graph_to_dict(G: nx.DiGraph) -> dict:
        nodes = [
            {"key": n, "node_type": d.get("node_type"), "node_id": d.get("node_id"), "label": d.get("label")}
            for n, d in G.nodes(data=True)
        ]
        edges = [
            {"source": u, "target": v, "edge_type": d.get("edge_type")}
            for u, v, d in G.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}
