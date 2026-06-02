"""One-shot Docker init: create tables + run demo."""
import asyncio
import sys
import os

sys.path.insert(0, "/app")

from db.base import create_all_tables
from store.metadata import MetadataStore
from store.blob import BlobStore
from core.lineage import LineageGraph
from db.base import AsyncSessionLocal


async def main():
    print("=== Pipeline Lineage Tracker — Docker Init ===")

    print("\n[1/3] Creating database tables...")
    await create_all_tables()
    print("  OK Tables created")

    print("\n[2/3] Running heart disease demo...")
    # Import and run the demo
    from examples.heart_disease.demo import run_demo
    await run_demo()

    print("\n[3/3] Summary...")
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        lg = LineageGraph(meta)
        stats = await lg.get_stats()

    print(f"  Dataset versions: {stats['dataset_versions']}")
    print(f"  Pipeline runs:    {stats['pipeline_runs']}")
    print(f"  Models:           {stats['models']}")
    print(f"  Lineage edges:    {stats['total_edges']}")
    print(f"  Graph depth avg:  {stats['avg_lineage_depth']}")

    print("\n=== Init complete. UI available at http://localhost:8502 ===")


if __name__ == "__main__":
    asyncio.run(main())
