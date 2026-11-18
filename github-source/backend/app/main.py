from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .config import get_settings


settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    from .services.personalization import refresh_arxiv_follows

    stop = threading.Event()

    def refresh_loop() -> None:
        while not stop.is_set():
            try:
                refresh_arxiv_follows()
            except Exception:
                logging.getLogger(__name__).exception("Scheduled arXiv refresh failed")
            stop.wait(max(300, settings.arxiv_refresh_interval_seconds))

    worker = threading.Thread(target=refresh_loop, name="arxiv-follow-refresh", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=2)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Phase 1 service and model integration scaffold.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.include_router(router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.app_name, "docs": "/docs"}
