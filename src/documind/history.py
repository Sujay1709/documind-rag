"""Persistent history of past interactions.

Each interaction records the question asked, the generated answer, the source
chunks it was grounded in, and a timestamp. History is stored as a JSON file so
it survives across app restarts. The store is intentionally simple (a single
JSON array) — fine for a local, single-user app.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings, get_settings
from .reranker import RankedChunk

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    """A single recorded question/answer interaction."""

    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def documents(self) -> list[str]:
        """Distinct source document names referenced by this entry."""
        return sorted({s.get("source", "unknown") for s in self.sources})


def _sources_from_chunks(chunks: list[RankedChunk]) -> list[dict]:
    return [
        {
            "source": c.metadata.get("source", "unknown"),
            "page": c.metadata.get("page"),
            "score": round(float(c.score), 4),
            "snippet": c.text[:500],
        }
        for c in chunks
    ]


def _path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.history_file


def load(settings: Settings | None = None) -> list[HistoryEntry]:
    """Load all history entries, newest first. Returns [] if none/corrupt."""
    path = _path(settings)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read history file %s: %s", path, exc)
        return []
    entries = [HistoryEntry(**item) for item in raw]
    return sorted(entries, key=lambda e: e.timestamp, reverse=True)


def _write(entries: list[HistoryEntry], settings: Settings | None = None) -> None:
    path = _path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(e) for e in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add(
    question: str,
    answer: str,
    chunks: list[RankedChunk] | None = None,
    settings: Settings | None = None,
) -> HistoryEntry:
    """Append a new interaction and persist it. Returns the stored entry."""
    settings = settings or get_settings()
    entry = HistoryEntry(
        question=question,
        answer=answer,
        sources=_sources_from_chunks(chunks or []),
    )
    # load() returns newest-first; keep chronological order on disk.
    existing = sorted(load(settings), key=lambda e: e.timestamp)
    existing.append(entry)
    if len(existing) > settings.max_history:
        existing = existing[-settings.max_history :]
    _write(existing, settings)
    logger.info("Recorded history entry %s", entry.id)
    return entry


def clear(settings: Settings | None = None) -> None:
    """Delete all stored history."""
    path = _path(settings)
    if path.exists():
        path.unlink()
    logger.info("Cleared history")
