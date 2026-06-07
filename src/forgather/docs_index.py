"""Vector index for the optional docs hybrid search.

Builds sentence-embeddings of heading-aware markdown chunks into a small
on-disk index (``docs/.built/.vector/``: ``meta.json`` + ``vectors.npy``), and
loads/queries them with cosine similarity. Built by ``forgather docs index``;
queried at runtime by ``tools/forgather_server/docs_vector.py``.

sentence-transformers is imported lazily (only when an embedding is actually
needed) so importing this module — e.g. to load chunk metadata or to chunk
text — never pulls in torch/transformers. numpy is already a core dependency.

The index is portable: it stores page-relative paths (not absolute), so an
index built in a container works wherever the repo lives.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .docs_corpus import BUILT_SUBDIR, iter_doc_files

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_VECTOR_SUBDIR = ".vector"
_META_NAME = "meta.json"
_VECTORS_NAME = "vectors.npy"
_MAX_DOC_BYTES = 25 * 1024 * 1024
# Chunk sizing (characters). Sections longer than this are split at paragraph
# boundaries so each embedded chunk stays focused.
_MAX_CHUNK_CHARS = 1200


def index_dir(repo_root: Path) -> Path:
    return Path(repo_root) / "docs" / BUILT_SUBDIR / _VECTOR_SUBDIR


# ---- chunking --------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    rel: str  # page path relative to docs/ (or repo-relative for CLAUDE.*)
    heading: str  # heading breadcrumb, e.g. "Trainers > AbstractTrainer"
    text: str  # chunk body (what's shown as the excerpt)
    order: int  # position within the page

    def embed_text(self) -> str:
        # Prefix the heading breadcrumb so the embedding has page context.
        return f"{self.heading}\n\n{self.text}" if self.heading else self.text


def _split_long(body: str, max_chars: int) -> List[str]:
    if len(body) <= max_chars:
        return [body]
    out: List[str] = []
    buf: List[str] = []
    size = 0
    for para in body.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if size + len(para) > max_chars and buf:
            out.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 2
    if buf:
        out.append("\n\n".join(buf))
    return out or [body[:max_chars]]


def chunk_markdown(
    rel: str, text: str, *, max_chars: int = _MAX_CHUNK_CHARS
) -> List[Chunk]:
    """Split markdown into heading-aware chunks (one+ per section)."""
    stack: List[Tuple[int, str]] = []  # (level, title)
    body_lines: List[str] = []
    chunks: List[Chunk] = []
    order = 0

    def breadcrumb() -> str:
        return " > ".join(title for _, title in stack)

    def flush():
        nonlocal order, body_lines
        body = "\n".join(body_lines).strip()
        body_lines = []
        if not body:
            return
        for piece in _split_long(body, max_chars):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(rel=rel, heading=breadcrumb(), text=piece, order=order))
                order += 1

    in_fence = False
    for line in text.splitlines():
        stripped = line.lstrip()
        # Toggle fenced-code state; a `#` line inside a fence is a comment, not
        # a heading, and must not split the chunk or pollute the breadcrumb.
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            body_lines.append(line)
            continue
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            flush()  # close the section that was accumulating
            level = len(m.group(1))
            title = m.group(2).strip()
            # Pop deeper-or-equal headings, then push this one.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body_lines.append(line)
    flush()
    return chunks


# ---- embedder (lazy sentence-transformers) ---------------------------------


class Embedder:
    """Lazy sentence-transformers wrapper. Normalizes for cosine via dot."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - dep is declared
                raise RuntimeError(
                    "docs vector search needs sentence-transformers "
                    "(a declared dependency); install it to build/query the index."
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._ensure().get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._ensure().encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype="float32")


# ---- build -----------------------------------------------------------------


@dataclass
class BuildReport:
    out_dir: Path
    model_name: str
    pages: int
    chunks: int
    fingerprint: str
    cleaned: bool = False


def _fingerprint(model_name: str, files: List[Tuple[Path, str]]) -> str:
    """Stable digest of (model + each page's rel/size/mtime) for staleness."""
    h = hashlib.sha256()
    h.update(model_name.encode())
    for path, rel in sorted(files, key=lambda t: t[1]):
        try:
            st = path.stat()
            h.update(f"\n{rel}:{st.st_size}:{int(st.st_mtime)}".encode())
        except OSError:
            h.update(f"\n{rel}:missing".encode())
    return h.hexdigest()


def _gather_chunks(repo_root: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    # Build over the FULL corpus (agent docs included); the runtime query
    # filters CLAUDE.* out for the user-facing endpoint.
    for path, rel in iter_doc_files(repo_root, include_agent_docs=True):
        try:
            if path.stat().st_size > _MAX_DOC_BYTES:
                continue
            text = path.read_text(errors="replace")
        except OSError:
            continue
        chunks.extend(chunk_markdown(rel, text))
    return chunks


def build_index(
    repo_root: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    embedder: Optional[Embedder] = None,
    clean: bool = False,
) -> BuildReport:
    """Embed all doc chunks and write the index to ``docs/.built/.vector/``.

    ``embedder`` is injectable for testing; defaults to a real (lazy) one.
    """
    repo_root = Path(repo_root)
    out = index_dir(repo_root)
    if clean and out.exists():
        for f in (out / _META_NAME, out / _VECTORS_NAME):
            f.unlink(missing_ok=True)

    files = iter_doc_files(repo_root, include_agent_docs=True)
    fingerprint = _fingerprint(model_name, files)
    chunks = _gather_chunks(repo_root)
    emb = embedder or Embedder(model_name)

    if chunks:
        vectors = emb.embed([c.embed_text() for c in chunks])
    else:
        vectors = np.zeros((0, emb.dim), dtype="float32")

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / _VECTORS_NAME, vectors)
    meta = {
        "model": model_name,
        "dim": int(vectors.shape[1]) if vectors.size else emb.dim,
        "count": len(chunks),
        "fingerprint": fingerprint,
        "chunks": [
            {"rel": c.rel, "heading": c.heading, "text": c.text, "order": c.order}
            for c in chunks
        ],
    }
    (out / _META_NAME).write_text(json.dumps(meta))
    pages = len({c.rel for c in chunks})
    return BuildReport(
        out_dir=out,
        model_name=model_name,
        pages=pages,
        chunks=len(chunks),
        fingerprint=fingerprint,
        cleaned=clean,
    )


def is_stale(repo_root: Path, *, model_name: str = DEFAULT_MODEL) -> bool:
    """True if no index exists or the corpus/model fingerprint changed."""
    loaded = load_index(repo_root)
    if loaded is None:
        return True
    files = iter_doc_files(Path(repo_root), include_agent_docs=True)
    return loaded.fingerprint != _fingerprint(model_name, files)


# ---- load + query ----------------------------------------------------------


@dataclass
class LoadedIndex:
    model_name: str
    dim: int
    fingerprint: str
    chunks: List[Dict[str, Any]]
    vectors: np.ndarray  # [n, dim], L2-normalized


def load_index(repo_root: Path) -> Optional[LoadedIndex]:
    out = index_dir(Path(repo_root))
    meta_path = out / _META_NAME
    vec_path = out / _VECTORS_NAME
    if not (meta_path.is_file() and vec_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        vectors = np.load(vec_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return LoadedIndex(
        model_name=meta.get("model", DEFAULT_MODEL),
        dim=int(meta.get("dim", vectors.shape[1] if vectors.size else 0)),
        fingerprint=meta.get("fingerprint", ""),
        chunks=meta.get("chunks", []),
        vectors=np.asarray(vectors, dtype="float32"),
    )


def query(idx: LoadedIndex, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    """Top-k (chunk_index, cosine_score) for a normalized query vector."""
    if idx.vectors.size == 0:
        return []
    sims = idx.vectors @ np.asarray(query_vec, dtype="float32")
    k = min(top_k, sims.shape[0])
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in top]
