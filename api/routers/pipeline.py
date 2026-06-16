import subprocess
import sys
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

router = APIRouter()

# ── Command registry ──────────────────────────────────────────────────────────

COMMANDS: Dict[str, Dict[str, Any]] = OrderedDict([
    ("ingest", {
        "label":       "Ingest News Posts",
        "description": "Load posts from data/posts/ into Qdrant (incremental — skips existing).",
        "cmd":         ["python3", "app/cli.py", "ingest"],
        "danger":      False,
    }),
    ("ingest_dry", {
        "label":       "Dry Run (News)",
        "description": "Parse + validate posts without writing to Qdrant.",
        "cmd":         ["python3", "app/cli.py", "ingest", "--dry-run"],
        "danger":      False,
    }),
    ("rebuild", {
        "label":       "Rebuild Index",
        "description": "Drop and recreate the news collection, then ingest all posts. Destructive.",
        "cmd":         ["python3", "app/cli.py", "rebuild-index"],
        "danger":      True,
    }),
    ("ingest_knowledge", {
        "label":       "Ingest Knowledge Base",
        "description": "Recreate the knowledge collection and load all data/knowledge/ docs. Destructive.",
        "cmd":         ["python3", "app/cli.py", "ingest-knowledge", "--recreate"],
        "danger":      True,
    }),
    ("evaluate", {
        "label":       "Evaluate News Retrieval",
        "description": "Run the news retrieval eval suite against tests/retrieval_eval.csv.",
        "cmd":         ["python3", "app/cli.py", "evaluate", "--verbose"],
        "danger":      False,
    }),
    ("evaluate_knowledge", {
        "label":       "Evaluate Knowledge Retrieval",
        "description": "Run the knowledge retrieval eval suite against tests/knowledge_eval.csv.",
        "cmd":         ["python3", "app/cli.py", "evaluate-knowledge", "--verbose"],
        "danger":      False,
    }),
])

# ── In-memory run store (last 50 runs) ────────────────────────────────────────

MAX_RUNS = 50
_runs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _store_run(run: Dict[str, Any]) -> None:
    _runs[run["run_id"]] = run
    while len(_runs) > MAX_RUNS:
        _runs.popitem(last=False)


# ── Pydantic models ───────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    command: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_summary(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id":      run["run_id"],
        "command":     run["command"],
        "label":       run["label"],
        "status":      run["status"],
        "started_at":  run["started_at"],
        "finished_at": run["finished_at"],
        "exit_code":   run["exit_code"],
        "line_count":  len(run["lines"]),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/commands")
def list_commands() -> Dict[str, Any]:
    return {
        "commands": [
            {
                "key":         k,
                "label":       v["label"],
                "description": v["description"],
                "danger":      v["danger"],
            }
            for k, v in COMMANDS.items()
        ]
    }


@router.get("")
def list_runs() -> List[Dict[str, Any]]:
    return [_run_summary(r) for r in reversed(list(_runs.values()))]


@router.get("/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return {
        **_run_summary(run),
        "lines": run["lines"],
    }


@router.post("")
def start_run(body: RunRequest) -> Dict[str, Any]:
    cmd_cfg = COMMANDS.get(body.command)
    if not cmd_cfg:
        raise HTTPException(400, f"Unknown command: {body.command}")

    for r in _runs.values():
        if r["status"] == "running":
            raise HTTPException(409, "Another run is already in progress. Wait for it to finish or cancel it.")

    run_id = str(uuid.uuid4())
    run: Dict[str, Any] = {
        "run_id":      run_id,
        "command":     body.command,
        "label":       cmd_cfg["label"],
        "status":      "running",
        "started_at":  datetime.utcnow().isoformat(),
        "finished_at": None,
        "exit_code":   None,
        "lines":       [],
        "_process":    None,
    }
    _store_run(run)

    def _worker() -> None:
        try:
            proc = subprocess.Popen(
                cmd_cfg["cmd"],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=None,  # inherit shell env (ANTHROPIC_API_KEY, PATH, etc.)
            )
            run["_process"] = proc
            for line in proc.stdout:  # type: ignore[union-attr]
                run["lines"].append(line.rstrip("\n"))
            proc.wait()
            run["exit_code"] = proc.returncode
            run["status"]    = "done" if proc.returncode == 0 else "error"
        except Exception as exc:
            run["lines"].append(f"[runner] {exc}")
            run["status"] = "error"
        finally:
            run["finished_at"] = datetime.utcnow().isoformat()
            run["_process"]    = None

    threading.Thread(target=_worker, daemon=True).start()
    return {"run_id": run_id, "status": "running"}


@router.post("/{run_id}/cancel")
def cancel_run(run_id: str) -> Dict[str, Any]:
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    proc = run.get("_process")
    if proc is not None and run["status"] == "running":
        proc.terminate()
        run["status"]      = "cancelled"
        run["finished_at"] = datetime.utcnow().isoformat()
    return {"ok": True, "run_id": run_id, "status": run["status"]}
