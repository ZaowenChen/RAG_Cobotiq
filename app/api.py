"""FastAPI application entrypoint."""

from fastapi import FastAPI

from retrieve import hybrid, rerank
from retrieve.policy import load_policy

app = FastAPI(title="Robot RAG")
POLICY = load_policy()


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
    return {"results": results}
