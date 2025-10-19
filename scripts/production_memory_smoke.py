#!/usr/bin/env python3
"""Smoke test harness for LangGraph production memory backends.

Run this script after spinning up your databases to confirm credentials work:

    POSTGRES_URI=postgresql://postgres:postgres@localhost:5442/postgres \
    MONGODB_URI="mongodb://localhost:27017" \
    REDIS_URI="redis://localhost:6379" \
    uv run python scripts/production_memory_smoke.py

Set `SMOKE_INITIALIZE=true` to invoke `.setup()` on supported backends during the run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.langgraph_learning.stage06_production_systems.agent_with_production_memory import (
    BACKENDS,
)


CHECKS: Iterable[tuple[str, str]] = (
    ("postgres", "POSTGRES_URI"),
    ("mongodb", "MONGODB_URI"),
    ("redis", "REDIS_URI"),
)


def main() -> None:
    initialize = os.environ.get("SMOKE_INITIALIZE", "false").lower() == "true"
    failures: list[str] = []

    for backend_key, env_var in CHECKS:
        uri = os.environ.get(env_var)
        backend = BACKENDS[backend_key]
        if not uri:
            print(f"[skip] {backend.name}: set {env_var} to enable this check.")
            continue

        print(f"[info] Connecting to {backend.name} at {uri!r}")
        try:
            with backend.sync_factory(uri, initialize) as (store, saver):
                store_cls = store.__class__.__name__
                saver_cls = saver.__class__.__name__
                print(
                    f"[pass] {backend.name}: store={store_cls}, checkpointer={saver_cls}"
                )
        except ModuleNotFoundError as exc:
            print(
                f"[warn] {backend.name}: missing optional dependency ({exc}). "
                "Install the appropriate langgraph-checkpoint package."
            )
            failures.append(backend.name)
        except Exception as exc:  # pragma: no cover - diagnostics path
            failures.append(backend.name)
            print(f"[fail] {backend.name}: {exc}")

    print("\nSmoke test complete.")
    if failures:
        print(f"{len(failures)} backend(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("All configured backends connected successfully.")


if __name__ == "__main__":
    main()
