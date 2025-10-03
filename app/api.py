"""FastAPI application entrypoint."""

from pathlib import Path

from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from retrieve import hybrid, rerank
from retrieve.policy import load_policy
from qa.generate import generate_answer
from ingestion.utils import media_root

app = FastAPI(title="Robot RAG")
POLICY = load_policy()
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "ui" / "templates")
MEDIA_ROOT = media_root()

app.mount(
    "/ui/static",
    StaticFiles(directory=BASE_DIR / "ui" / "static"),
    name="ui-static",
)

app.mount(
    "/media",
    StaticFiles(directory=MEDIA_ROOT),
    name="media",
)


def _decorate_media_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    media_path = entry.get("media_path")
    if media_path:
        entry["media_url"] = f"/media/{media_path}"
    return entry


def _attach_media_urls(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_decorate_media_entry(dict(item)) for item in items]


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple healthcheck payload."""
    return {"status": "ok"}


@app.get("/search")
def search(query: str, audience_level: str | None = None, robot_model: str | None = None) -> dict[str, object]:
    """Execute hybrid search followed by reranking when enabled."""
    filters: dict[str, str] = {}
    if audience_level:
        filters["audience_level"] = audience_level
    if robot_model:
        filters["robot_model"] = robot_model

    results = hybrid.retrieve(query, filters)
    if POLICY.get("rerank", {}).get("enabled", False):
        results = rerank.rerank(query, results)

    answer_payload = generate_answer(query, results)

    response: dict[str, object] = {
        "results": _attach_media_urls(results),
        "figures": _attach_media_urls(answer_payload.get("figures", [])),
    }

    for key, value in answer_payload.items():
        if key == "figures":
            continue
        response[key] = value
    return response


@app.get("/", response_class=HTMLResponse)
def chat_ui(request: Request) -> HTMLResponse:
    """Serve the lightweight chat interface."""
    return templates.TemplateResponse("chat.html", {"request": request})
