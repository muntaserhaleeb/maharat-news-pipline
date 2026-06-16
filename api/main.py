import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Project root must be on sys.path before any service imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import init_db
from api.routers import config, dashboard, entities, generator, pipeline, retrieval, review, taxonomy


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Maharat RAG Operations UI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router,   prefix="/api/dashboard",  tags=["dashboard"])
app.include_router(config.router,      prefix="/api/config",     tags=["config"])
app.include_router(retrieval.router,   prefix="/api/retrieval",  tags=["retrieval"])
app.include_router(generator.router,   prefix="/api/generator",  tags=["generator"])
app.include_router(entities.router,    prefix="/api/entities",   tags=["entities"])
app.include_router(taxonomy.router,    prefix="/api/taxonomy",   tags=["taxonomy"])
app.include_router(pipeline.router,    prefix="/api/pipeline",   tags=["pipeline"])
app.include_router(review.router,      prefix="/api/review",     tags=["review"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
