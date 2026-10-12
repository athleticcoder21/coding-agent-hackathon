"""Build a compact, reusable index over the large competition artifacts."""

from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

FILE_HEADER = re.compile(r"(?m)^\+\+\+ b/(.+)$")


def _title(problem: str) -> str:
    for line in problem.splitlines():
        line = line.strip().lstrip("# ")
        if line:
            return line[:160]
    return "Untitled issue"


def _changed_files(patch: str) -> list[str]:
    return list(dict.fromkeys(FILE_HEADER.findall(patch)))


def build_catalog(data_dir: Path) -> dict:
    task_path = data_dir / "tasks.jsonl"
    if not task_path.is_file():
        raise FileNotFoundError(f"Competition task list missing: {task_path}")
    tasks = []
    artifacts: dict[str, dict] = {}
    for line in task_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stem = f"{row['repo'].split('/')[-1]}_{row['base_commit']}"
        snapshot = data_dir / "snapshots" / f"{row['instance_id']}.tgz"
        tasks.append({
            "id": row["instance_id"],
            "repo": row["repo"],
            "repo_short": row["repo"].split("/")[-1],
            "base_commit": row["base_commit"],
            "artifact": stem,
            "created_at": row.get("created_at"),
            "title": _title(row.get("problem_statement") or ""),
            "changed_files": _changed_files(row.get("patch") or ""),
            "test_files": _changed_files(row.get("test_patch") or ""),
            "snapshot_bytes": snapshot.stat().st_size if snapshot.is_file() else None,
            "patch_bytes": len((row.get("patch") or "").encode()),
        })
        if stem not in artifacts:
            graph_path = data_dir / "graphs" / f"{stem}.json"
            embedding_path = data_dir / "embeddings" / f"{stem}.npz"
            info = {"stem": stem, "repo": row["repo"], "base_commit": row["base_commit"],
                    "tasks": [], "graph_bytes": None, "embedding_bytes": None,
                    "nodes": None, "edges": None, "vectors": None, "dimensions": None,
                    "edge_types": {}}
            if graph_path.is_file():
                graph = json.loads(graph_path.read_text())
                info["graph_bytes"] = graph_path.stat().st_size
                info["nodes"] = len(graph.get("nodes", []))
                info["edges"] = len(graph.get("edges", []))
                info["edge_types"] = dict(Counter(edge.get("type", "unknown")
                                                  for edge in graph.get("edges", [])))
            if embedding_path.is_file():
                with np.load(embedding_path, allow_pickle=False) as archive:
                    info["vectors"] = len(archive.files)
                    info["dimensions"] = int(archive[archive.files[0]].shape[0]) if archive.files else 0
                info["embedding_bytes"] = embedding_path.stat().st_size
            artifacts[stem] = info
        artifacts[stem]["tasks"].append(row["instance_id"])

    files = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(data_dir).as_posix()
        if relative == "gemma-4-developer-agent.zip":
            kind = "archive"
        elif "/" in relative:
            kind = relative.split("/", 1)[0]
        else:
            kind = "guide" if relative.endswith(".md") else "tasks"
        files.append({"path": relative, "kind": kind, "bytes": path.stat().st_size})
    repos = Counter(task["repo_short"] for task in tasks)
    return {
        "tasks": tasks,
        "artifacts": artifacts,
        "files": files,
        "summary": {
            "task_count": len(tasks),
            "repo_count": len(repos),
            "repos": dict(repos),
            "artifact_count": len(artifacts),
            "graph_count": sum(a["nodes"] is not None for a in artifacts.values()),
            "embedding_count": sum(a["vectors"] is not None for a in artifacts.values()),
            "snapshot_count": sum(t["snapshot_bytes"] is not None for t in tasks),
            "total_nodes": sum(a["nodes"] or 0 for a in artifacts.values()),
            "total_edges": sum(a["edges"] or 0 for a in artifacts.values()),
            "total_vectors": sum(a["vectors"] or 0 for a in artifacts.values()),
            "file_count": len(files),
        },
    }


def load_catalog(data_dir: Path, index_path: Path) -> dict:
    source = data_dir / "tasks.jsonl"
    latest_source = max((path.stat().st_mtime_ns for path in data_dir.rglob("*")),
                        default=source.stat().st_mtime_ns)
    if index_path.is_file() and index_path.stat().st_mtime_ns >= latest_source:
        return json.loads(index_path.read_text())
    catalog = build_catalog(data_dir)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(catalog, ensure_ascii=False, separators=(",", ":")))
    return catalog
