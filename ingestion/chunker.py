"""Chunker — Phase 1 (Data Ingestion).

Splits the raw Wikipedia text in ``data/raw/`` into overlapping,
embedding-sized chunks and writes them to ``data/processed/``.

Responsibilities:
- Read each raw article and clean it (strip markup, references, etc.).
- Split text into token-bounded chunks with configurable overlap
  (see CHUNK_SIZE / CHUNK_OVERLAP in ``config.py``).
- Assign every chunk a STABLE, unique ``chunk_id`` plus source
  metadata (article title, source URL, char offsets). This chunk_id
  is what every downstream fact must cite (Critical Rule #2).
- Persist the processed chunks (text + metadata) to ``data/processed/``
  in a format ready for ``embedder.py``.

Implementation notes:
- Token counting uses ``tiktoken``'s ``cl100k_base`` as a model-agnostic sizing
  heuristic. The embedding model is now the local sentence-transformer
  ``all-mpnet-base-v2`` (384-token window), which tiktoken doesn't recognise, so
  the encoder lookup falls back to cl100k. cl100k and the model's own tokenizer
  don't count identically, so the default ``CHUNK_SIZE`` of 300 leaves a safe
  buffer under that 384-token limit.
- Chunking is paragraph-aware: whole paragraphs are packed together up to
  ``chunk_size`` tokens, and the overlap carries whole trailing paragraphs
  into the next chunk. We only ever split *within* a paragraph (on sentence
  boundaries, then — as a last resort for a single oversized sentence — on
  raw tokens), so chunks don't cut mid-sentence except when unavoidable.
- The public ``chunk_documents`` takes the LangChain ``Document``s produced by
  ``wikipedia_loader.load_topics``. ``load_raw_documents`` reconstructs those
  Documents from the ``data/raw/`` cache so the chunker can run offline.
"""

from __future__ import annotations

import json
import logging
import re

import tiktoken
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

# Trailing Wikipedia sections that carry no historical prose; everything from
# the first such standalone heading onward is dropped during cleaning.
_BOILERPLATE_HEADINGS = frozenset(
    {
        "see also",
        "references",
        "notes",
        "citations",
        "sources",
        "further reading",
        "external links",
        "bibliography",
        "footnotes",
    }
)

# Sentence boundary: end punctuation followed by whitespace. Only used to break
# a single paragraph that is itself larger than a whole chunk.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Filename sanitisation — kept byte-for-byte in step with
# ``wikipedia_loader._safe_filename`` so a chunk's slug matches its raw file.
_INVALID_FS_CHARS = re.compile(r'[\\/:*?"<>|]+')
_WHITESPACE = re.compile(r"\s+")

_encoder: tiktoken.Encoding | None = None


class Chunk(BaseModel):
    """One embedding-sized slice of an article, with its citation metadata.

    ``chunk_id`` is the stable handle every downstream verified fact must cite
    (Critical Rule #2). ``start_char`` / ``end_char`` are best-effort offsets
    into the *cleaned* article text (overlapping chunks share characters).
    """

    chunk_id: str
    text: str
    source_title: str
    source_url: str
    token_count: int
    start_char: int
    end_char: int
    page_id: int | None = None
    chunk_index: int = Field(..., ge=0)


def _get_encoder() -> tiktoken.Encoding:
    """Return (and cache) the tiktoken encoding for the embedding model.

    Falls back to ``cl100k_base`` if the model name isn't recognised — which is
    the case for the local sentence-transformer ``all-mpnet-base-v2``, so the
    counts are a cl100k heuristic used purely for chunk sizing. The first call may
    fetch the BPE vocab over the network; tiktoken caches it on disk thereafter.
    """
    global _encoder
    if _encoder is None:
        try:
            _encoder = tiktoken.encoding_for_model(settings.EMBEDDING_MODEL)
        except KeyError:
            _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def _slug(title: str) -> str:
    """Filesystem/Chroma-safe stem for a title (matches the raw loader's slug)."""
    cleaned = _INVALID_FS_CHARS.sub("", title.strip())
    cleaned = _WHITESPACE.sub("_", cleaned)
    cleaned = cleaned.strip("._")
    return (cleaned.lower() or "untitled")[:120]


def _clean_text(text: str) -> str:
    """Drop trailing boilerplate sections and normalise whitespace.

    Truncates at the first standalone line that is a known boilerplate heading
    (References, See also, ...), trims trailing whitespace per line, and removes
    blank lines. WIKI-format extracts are otherwise already plain text.
    """
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in _BOILERPLATE_HEADINGS:
            break  # everything from here on is reference/appendix material
        if stripped:
            kept.append(stripped)
    return "\n".join(kept)


def _paragraphs(text: str) -> list[str]:
    """Split cleaned text into atomic paragraph units (one per non-empty line)."""
    return [line for line in (l.strip() for l in text.split("\n")) if line]


def _token_slice(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Hard-split text on raw token windows — last resort for an oversized unit.

    This is the only path that may cut mid-sentence; reached only when a single
    sentence exceeds ``chunk_size`` tokens.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    step = max(1, chunk_size - overlap)
    pieces: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        pieces.append(enc.decode(window).strip())
        if start + chunk_size >= len(tokens):
            break
    return [p for p in pieces if p]


def _atomic_units(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    """Break a paragraph that alone exceeds ``chunk_size`` into packable units.

    Prefers sentence boundaries; only a single sentence larger than a whole
    chunk is token-sliced.
    """
    if _count_tokens(paragraph) <= chunk_size:
        return [paragraph]

    units: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        if _count_tokens(sentence) <= chunk_size:
            units.append(sentence)
        else:
            units.extend(_token_slice(sentence, chunk_size, overlap))
    return units


def _overlap_tail(units: list[str], overlap: int) -> list[str]:
    """Trailing whole units whose combined tokens stay within ``overlap``."""
    tail: list[str] = []
    tail_tokens = 0
    for unit in reversed(units):
        unit_tokens = _count_tokens(unit)
        if tail and tail_tokens + unit_tokens > overlap:
            break
        tail.insert(0, unit)
        tail_tokens += unit_tokens
    return tail


def _pack(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack whole units into ``chunk_size``-token chunks with overlap.

    When the next unit won't fit, the current chunk is flushed and the next one
    is seeded with trailing whole units totalling roughly ``overlap`` tokens, so
    overlap never cuts mid-sentence. Fit is measured on the *joined* chunk text
    (counting the newline separators), so an emitted chunk never exceeds
    ``chunk_size`` tokens; carried units are dropped if needed to make room for
    the incoming unit (which is itself <= ``chunk_size`` via ``_atomic_units``).
    """
    # Expand any unit larger than a whole chunk into smaller packable units.
    expanded: list[str] = []
    for unit in units:
        expanded.extend(_atomic_units(unit, chunk_size, overlap))

    chunks: list[str] = []
    current: list[str] = []

    for unit in expanded:
        if current and _count_tokens("\n".join((*current, unit))) > chunk_size:
            chunks.append("\n".join(current))
            current = _overlap_tail(current, overlap)
            while current and _count_tokens("\n".join((*current, unit))) > chunk_size:
                current.pop(0)
        current.append(unit)

    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_documents(
    docs: list[Document],
    *,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split loader Documents into token-bounded, overlapping ``Chunk``s.

    Each document is cleaned (boilerplate sections dropped), split on paragraph
    boundaries, and packed into chunks of at most ``chunk_size`` tokens with
    ``overlap`` tokens of whole-paragraph context carried between consecutive
    chunks. Empty / whitespace-only documents yield no chunks.

    Args:
        docs: Documents from ``wikipedia_loader.load_topics`` (or
            ``load_raw_documents``); metadata carries ``title`` / ``url`` /
            ``page_id``.
        chunk_size: Max tokens per chunk (defaults to ``settings.CHUNK_SIZE``).
        overlap: Approx. token overlap between consecutive chunks
            (defaults to ``settings.CHUNK_OVERLAP``).

    Returns:
        Chunks across all documents, each with a stable ``chunk_id`` of the form
        ``"<slug>_<index>"`` and its source/citation metadata.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

    all_chunks: list[Chunk] = []
    for doc in docs:
        meta = doc.metadata or {}
        title = meta.get("title") or meta.get("requested_topic") or "untitled"
        url = meta.get("url", "")
        page_id = meta.get("page_id")
        slug = _slug(title)

        cleaned = _clean_text(doc.page_content or "")
        if not cleaned:
            logger.warning("Skipping %r: no text after cleaning", title)
            continue

        texts = _pack(_paragraphs(cleaned), chunk_size, overlap)

        # Best-effort char offsets via a forward-only cursor over cleaned text.
        cursor = 0
        for index, text in enumerate(texts):
            head = text.split("\n", 1)[0]
            start = cleaned.find(head, cursor)
            if start == -1:
                start = cursor
            end = start + len(text)
            cursor = max(cursor, start + len(head))
            all_chunks.append(
                Chunk(
                    chunk_id=f"{slug}_{index}",
                    text=text,
                    source_title=title,
                    source_url=url,
                    token_count=_count_tokens(text),
                    start_char=start,
                    end_char=end,
                    page_id=page_id,
                    chunk_index=index,
                )
            )

    logger.info(
        "chunk_documents complete: %d document(s) -> %d chunk(s)",
        len(docs),
        len(all_chunks),
    )
    return all_chunks


def save_chunks(chunks: list[Chunk]) -> dict[str, int]:
    """Persist chunks to ``data/processed/<slug>.jsonl`` (one chunk per line).

    Chunks are grouped by article slug (the ``chunk_id`` prefix), mirroring the
    one-file-per-article layout of ``data/raw/``. Returns a ``{slug: count}`` map.
    """
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        slug = chunk.chunk_id.rsplit("_", 1)[0]
        grouped.setdefault(slug, []).append(chunk)

    counts: dict[str, int] = {}
    for slug, group in grouped.items():
        path = settings.PROCESSED_DIR / f"{slug}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for chunk in group:
                fh.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
        counts[slug] = len(group)
        logger.info("Saved %d chunk(s) -> %s", len(group), path.name)
    return counts


def load_raw_documents() -> list[Document]:
    """Rebuild Documents from the ``data/raw/`` cache (offline; no network)."""
    if not settings.RAW_DIR.exists():
        return []

    documents: list[Document] = []
    for json_path in sorted(settings.RAW_DIR.glob("*.json")):
        txt_path = json_path.with_suffix(".txt")
        if not txt_path.exists():
            logger.warning("Skipping %s: no matching .txt", json_path.name)
            continue
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        text = txt_path.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


if __name__ == "__main__":
    # Offline smoke test over the cached raw articles. Run from the project root:
    #   D:\historyos\venv\Scripts\python.exe -m ingestion.chunker
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    docs = load_raw_documents()
    chunks = chunk_documents(docs)
    counts = save_chunks(chunks)

    print(f"\nProcessed {len(docs)} document(s) -> {len(chunks)} chunk(s)")
    if chunks:
        token_counts = [c.token_count for c in chunks]
        print(
            f"  tokens/chunk: min={min(token_counts)} "
            f"avg={sum(token_counts) // len(token_counts)} "
            f"max={max(token_counts)} (limit {settings.CHUNK_SIZE})"
        )
        over_limit = [c.chunk_id for c in chunks if c.token_count > settings.CHUNK_SIZE]
        duplicate_ids = len(chunks) - len({c.chunk_id for c in chunks})
        print(f"  over-limit chunks: {len(over_limit)}  duplicate ids: {duplicate_ids}")
        print("  per article:")
        for slug, count in sorted(counts.items()):
            print(f"    - {slug}: {count}")
        print("  sample ids:", ", ".join(c.chunk_id for c in chunks[:5]))
