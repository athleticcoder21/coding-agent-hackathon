"""Read-only, loopback-only API for the competition data explorer."""

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zipfile

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from explorer.catalog import load_catalog

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "explorer" / "web"
HUNK_NAME = re.compile(r"(?m)^@@[^\n]*@@[^\n]*\b(?:def|class)\s+([A-Za-z_][A-Za-z_0-9]*)")


def _module_for(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    parts = list(Path(path).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts.pop(0)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or None


def create_app(data_dir: Path | None = None) -> FastAPI:
    data_dir = (data_dir or ROOT / "data" / "competition").resolve()
    default_data = (ROOT / "data" / "competition").resolve()
    index_name = "index.json" if data_dir == default_data else \
        f"index-{hashlib.sha256(str(data_dir).encode()).hexdigest()[:12]}.json"
    catalog = load_catalog(data_dir, ROOT / "data" / "explorer" / index_name)
    tasks = {row["id"]: row for row in catalog["tasks"]}
    raw_tasks = {row["instance_id"]: row for row in (
        json.loads(line) for line in (data_dir / "tasks.jsonl").read_text().splitlines() if line.strip()
    )}
    app = FastAPI(title="Gemma 4 Data Atlas", docs_url=None, redoc_url=None)

    def task_or_404(task_id: str) -> dict:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Unknown task")
        return task

    @lru_cache(maxsize=3)
    def graph_for(stem: str) -> dict:
        path = data_dir / "graphs" / f"{stem}.json"
        if not path.is_file():
            raise HTTPException(404, "Graph unavailable")
        raw = json.loads(path.read_text())
        nodes = {node["id"]: node for node in raw["nodes"]}
        incoming: dict[str, list] = defaultdict(list)
        outgoing: dict[str, list] = defaultdict(list)
        for edge in raw["edges"]:
            outgoing[edge["source"]].append(edge)
            incoming[edge["target"]].append(edge)
        return {"nodes": nodes, "edges": raw["edges"], "incoming": incoming,
                "outgoing": outgoing}

    @lru_cache(maxsize=2)
    def embeddings_for(stem: str):
        path = data_dir / "embeddings" / f"{stem}.npz"
        if not path.is_file():
            raise HTTPException(404, "Embeddings unavailable")
        with np.load(path, allow_pickle=False) as archive:
            names = archive.files
            matrix = np.stack([archive[name] for name in names]).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        normalized = matrix / np.maximum(norms[:, None], 1e-12)
        return names, {name: i for i, name in enumerate(names)}, matrix, normalized, norms

    @lru_cache(maxsize=3)
    def snapshot_members(task_id: str) -> dict[str, tuple[str, int]]:
        path = data_dir / "snapshots" / f"{task_id}.tgz"
        if not path.is_file():
            raise HTTPException(404, "Snapshot unavailable")
        found = {}
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                name = member.name[2:] if member.name.startswith("./") else member.name
                if name.startswith(".git/") or name.startswith(".venv/"):
                    continue
                found[name] = (member.name, member.size)
        return found

    @app.get("/api/catalog")
    def get_catalog():
        return catalog

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str):
        task = task_or_404(task_id)
        raw = raw_tasks[task_id]
        return {**task, "problem_statement": raw.get("problem_statement") or "",
                "hints_text": raw.get("hints_text") or "", "patch": raw.get("patch") or "",
                "test_patch": raw.get("test_patch") or "",
                "artifact_summary": catalog["artifacts"][task["artifact"]]}

    @app.get("/api/tasks/{task_id}/focus")
    def get_focus(task_id: str):
        task = task_or_404(task_id)
        graph = graph_for(task["artifact"])
        names = list(graph["nodes"])
        groups = []
        for path in task["changed_files"]:
            module = _module_for(path)
            matches = [name for name in names if module and
                       (name == module or name.startswith(module + "."))]
            groups.append({"path": path, "module": module, "count": len(matches),
                           "symbols": matches[:60]})
        hunk_names = HUNK_NAME.findall(raw_tasks[task_id].get("patch") or "")
        default = next((candidate for name in hunk_names for group in groups
                        for candidate in group["symbols"]
                        if candidate.rsplit(".", 1)[-1] == name), None)
        if default is None:
            default = next((name for group in groups for name in group["symbols"]), None)
        if default is None and names:
            default = max(names, key=lambda name: len(graph["incoming"][name]) +
                          len(graph["outgoing"][name]))
        return {"groups": groups, "default_node": default, "hunk_names": hunk_names}

    @app.get("/api/tasks/{task_id}/graph/search")
    def search_graph(task_id: str, q: str = "", limit: int = Query(40, ge=1, le=100)):
        task = task_or_404(task_id)
        graph = graph_for(task["artifact"])
        needle = q.strip().lower()
        matches = [name for name in graph["nodes"] if needle in name.lower()]
        matches.sort(key=lambda name: (not name.lower().endswith(needle), len(name), name))
        return {"total": len(matches), "results": [{"id": name,
                 "degree": len(graph["incoming"][name]) + len(graph["outgoing"][name])}
                for name in matches[:limit]]}

    @app.get("/api/tasks/{task_id}/graph/neighborhood")
    def neighborhood(task_id: str, node: str, depth: int = Query(1, ge=1, le=2),
                     limit: int = Query(75, ge=5, le=150)):
        task = task_or_404(task_id)
        graph = graph_for(task["artifact"])
        if node not in graph["nodes"]:
            raise HTTPException(404, "Symbol unavailable in this graph")
        selected = {node}
        frontier = {node}
        for _ in range(depth):
            adjacent = {edge["source"] for name in frontier for edge in graph["incoming"][name]}
            adjacent.update(edge["target"] for name in frontier for edge in graph["outgoing"][name])
            adjacent -= selected
            ranked = sorted(adjacent, key=lambda name: (-len(graph["incoming"][name]) -
                                                       len(graph["outgoing"][name]), name))
            frontier = set(ranked[:max(0, limit - len(selected))])
            selected.update(frontier)
            if not frontier or len(selected) >= limit:
                break
        inbound = {edge["source"] for edge in graph["incoming"][node]}
        outbound = {edge["target"] for edge in graph["outgoing"][node]}
        nodes = []
        for name in sorted(selected):
            role = "center" if name == node else "both" if name in inbound and name in outbound else \
                   "caller" if name in inbound else "callee" if name in outbound else "second"
            nodes.append({"id": name, "role": role,
                          "in_degree": len(graph["incoming"][name]),
                          "out_degree": len(graph["outgoing"][name])})
        edges = [{"source": edge["source"], "target": edge["target"],
                  "type": edge.get("type", "calls")}
                 for edge in graph["edges"] if edge["source"] in selected and edge["target"] in selected]
        return {"center": node, "nodes": nodes, "edges": edges,
                "direct_callers": len(inbound), "direct_callees": len(outbound),
                "limited": len(selected) >= limit}

    @app.get("/api/tasks/{task_id}/graph/node")
    def graph_node(task_id: str, node: str):
        task = task_or_404(task_id)
        graph = graph_for(task["artifact"])
        data = graph["nodes"].get(node)
        if data is None:
            raise HTTPException(404, "Symbol unavailable in this graph")
        return {"id": node, "name": data.get("name", node), "text": data.get("text", ""),
                "callers": [edge["source"] for edge in graph["incoming"][node]][:100],
                "callees": [edge["target"] for edge in graph["outgoing"][node]][:100]}

    @app.get("/api/tasks/{task_id}/embeddings/nearest")
    def nearest(task_id: str, node: str, k: int = Query(24, ge=3, le=60)):
        task = task_or_404(task_id)
        names, positions, matrix, normalized, norms = embeddings_for(task["artifact"])
        index = positions.get(node)
        if index is None:
            raise HTTPException(404, "Symbol unavailable in this embedding archive")
        scores = normalized @ normalized[index]
        order = np.argsort(-scores)[:k + 1]
        order = [int(i) for i in order if int(i) != index][:k]
        selection = [index] + order
        reduced = normalized[selection]
        centered = reduced - reduced.mean(axis=0)
        u, singular, _ = np.linalg.svd(centered, full_matrices=False)
        coords = u[:, :2] * singular[:2]
        scale = max(float(np.abs(coords).max()), 1e-9)
        coords = coords / scale
        points = [{"id": names[i], "similarity": round(float(scores[i]), 5),
                   "x": round(float(coords[j, 0]), 5), "y": round(float(coords[j, 1]), 5),
                   "norm": round(float(norms[i]), 5)} for j, i in enumerate(selection)]
        return {"query": node, "dimensions": int(matrix.shape[1]),
                "archive_vectors": len(names), "vector_preview":
                [round(float(x), 5) for x in matrix[index, :16]], "points": points,
                "projection": "PCA of the selected nearest vectors; cosine similarity determines rank"}

    @app.get("/api/tasks/{task_id}/snapshot")
    def get_snapshot(task_id: str, q: str = "", limit: int = Query(160, ge=1, le=500)):
        task_or_404(task_id)
        members = snapshot_members(task_id)
        needle = q.strip().lower()
        matches = [(path, size) for path, (_, size) in members.items() if needle in path.lower()]
        matches.sort()
        return {"total": len(matches), "files": [{"path": path, "bytes": size}
                                                   for path, size in matches[:limit]]}

    @app.get("/api/tasks/{task_id}/snapshot/file")
    def get_snapshot_file(task_id: str, path: str):
        task_or_404(task_id)
        member = snapshot_members(task_id).get(path)
        if member is None:
            raise HTTPException(404, "File unavailable in this snapshot")
        original, size = member
        if size > 250_000:
            return {"path": path, "bytes": size, "text": None, "reason": "File exceeds 250 KB preview limit"}
        with tarfile.open(data_dir / "snapshots" / f"{task_id}.tgz", "r:gz") as archive:
            stream = archive.extractfile(original)
            content = stream.read(250_001) if stream else b""
        binary = b"\x00" in content
        return {"path": path, "bytes": size, "text": None if binary else content.decode("utf-8", "replace"),
                "reason": "Binary file" if binary else None}

    @app.get("/api/files/preview")
    def preview_file(path: str):
        record = next((item for item in catalog["files"] if item["path"] == path), None)
        if record is None:
            raise HTTPException(404, "Unknown dataset file")
        target = data_dir / path
        if record["kind"] == "graphs":
            stem = target.stem
            return {**record, "summary": catalog["artifacts"].get(stem), "text": None}
        if record["kind"] == "embeddings":
            stem = target.stem
            return {**record, "summary": catalog["artifacts"].get(stem), "text": None}
        if record["kind"] == "wheels":
            with zipfile.ZipFile(target) as archive:
                names = archive.namelist()
                metadata = next((name for name in names if name.endswith(".dist-info/METADATA")), None)
                description = archive.read(metadata).decode("utf-8", "replace")[:4000] if metadata else ""
            return {**record, "entries": len(names), "text": description,
                    "sample_entries": names[:30]}
        if record["kind"] in {"snapshots", "archive"}:
            return {**record, "text": None, "reason": "Open a task to browse its snapshot"}
        if record["bytes"] > 250_000:
            return {**record, "text": None, "reason": "File exceeds 250 KB preview limit"}
        content = target.read_bytes()
        binary = b"\x00" in content
        return {**record, "text": None if binary else content.decode("utf-8", "replace"),
                "reason": "Binary file" if binary else None}

    @app.get("/api/health")
    def health():
        return {"ok": True, "tasks": len(tasks), "data_dir": str(data_dir)}

    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    return app
