# Maharat RAG Ops UI

Internal web UI for managing the Maharat RAG pipeline.  
Not connected to the live website — outputs stay local.

---

## Phase 1 — Config Management Foundation

| # | Feature | Status |
|---|---------|--------|
| 1 | Backend health endpoint | ✅ `GET /api/health` |
| 2 | Config file listing | ✅ `GET /api/config` |
| 3 | Config read | ✅ `GET /api/config/{name}` |
| 4 | Config validate | ✅ `POST /api/config/validate` |
| 5 | Config save with version backup | ✅ `PUT /api/config/{name}` |
| 6 | Sidebar navigation | ✅ Dashboard + Config Manager |
| 7 | Dashboard placeholder | ✅ System status + data counts |
| 8 | Config Manager (editor + validation) | ✅ Edit, Validate, Save, History, Restore |

---

## Running locally

### Prerequisites

```bash
# Python deps (one-time)
source .venv/bin/activate
pip3 install -r requirements.txt

# Frontend deps (one-time)
cd ui && npm install && cd ..
```

### Start (one command)

```bash
./run_ui.sh
```

Then open **http://localhost:5173**

### Start manually (two terminals)

**Terminal 1 — Backend**
```bash
source .venv/bin/activate
python3 -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Frontend**
```bash
cd ui && npm run dev
```

---

## API reference (Phase 1)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Health check → `{"status":"ok"}` |
| `GET`  | `/api/dashboard/status` | Phase 1 status — system, data counts, config files |
| `GET`  | `/api/config` | List all managed config files |
| `GET`  | `/api/config/{name}` | Read config — returns raw YAML + parsed dict |
| `POST` | `/api/config/validate` | Validate YAML content (syntax + structure) without saving |
| `PUT`  | `/api/config/{name}` | Save config (backs up current version first) |
| `GET`  | `/api/config/{name}/history` | Version history (last 30) |
| `POST` | `/api/config/{name}/rollback/{id}` | Restore a previous version |

### Validate request/response

```jsonc
// POST /api/config/validate
{ "content": "entities:\n  ...", "name": "entities.yaml" }

// Response
{ "valid": true, "errors": [], "warnings": [] }
// or
{ "valid": false, "errors": ["YAML syntax error …"], "warnings": [] }
// or (valid syntax, missing expected key)
{ "valid": true, "errors": [], "warnings": ["Expected top-level key(s) not found: entities"] }
```

---

## Managed config files

| File | Top-level keys checked |
|------|----------------------|
| `entities.yaml` | `entities` |
| `taxonomy.yaml` | `category_rules`, `tags` |
| `generation.yaml` | `generation` |
| `chunking.yaml` | `chunking` |
| `knowledge_chunking.yaml` | `chunking` |
| `qdrant.yaml` | `qdrant`, `collections` |
| `editorial_style.yaml` | _(no structural check)_ |

---

## Storage

| Path | Contents |
|------|----------|
| `storage/maharat_ops.db` | SQLite — `config_versions`, `pipeline_runs` |

The database is gitignored and created automatically on first startup.

---

## Project layout

```
api/
├── main.py                  # FastAPI app, lifespan, CORS, router mounting
├── db.py                    # SQLite init + connection helper
├── models/schemas.py        # Pydantic models
└── routers/
    ├── dashboard.py         # GET /api/dashboard/status
    └── config.py            # All /api/config/* endpoints

ui/
├── src/
│   ├── App.tsx              # BrowserRouter + routes
│   ├── api/client.ts        # fetch wrapper (proxied to :8000)
│   ├── components/Layout.tsx
│   └── pages/
│       ├── Dashboard.tsx    # System status + config file table
│       └── ConfigManager.tsx # YAML editor + validate + save + history

storage/
└── maharat_ops.db           # Created on startup (gitignored)
```

---

## Extending to Phase 2

Each new module needs four things:

1. Router in `api/routers/<name>.py`
2. Mount in `api/main.py`: `app.include_router(..., prefix="/api/<name>")`
3. Page in `ui/src/pages/<Name>.tsx`
4. Nav entry in `ui/src/components/Layout.tsx` (move from the "Phase 2" grayed list)
