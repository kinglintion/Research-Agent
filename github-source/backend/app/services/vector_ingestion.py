from __future__ import annotations

import uuid

from qdrant_client import models

from ..config import get_settings
from .connections import qdrant_client
from .model_adapters import get_embedder


def _chunks(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def index_paper_content(
    *,
    paper_id: str,
    version_id: str,
    title: str,
    markdown: str,
    source_filename: str,
    parser_name: str,
    pages: list[dict] | None = None,
    replace_existing: bool = True,
) -> dict:
    settings = get_settings()
    collection = settings.qdrant_papers_collection
    chunk_rows: list[tuple[str, int | None]] = []
    if pages:
        for page in pages:
            page_number = page.get("page")
            chunk_rows.extend((chunk, page_number) for chunk in _chunks(page.get("text", "")))
    if not chunk_rows:
        chunk_rows = [(chunk, None) for chunk in _chunks(markdown)]
    chunks = [row[0] for row in chunk_rows]
    if not chunks:
        return {"status": "skipped", "reason": "no_text_chunks", "chunks": 0}

    embedder = get_embedder()
    vectors = embedder.get_text_embedding_batch(chunks, show_progress=False)
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding model returned a different number of vectors than chunks")
    vector_size = len(vectors[0])
    if vector_size != settings.expected_embedding_dimension:
        raise RuntimeError(
            f"Embedding dimension mismatch: got {vector_size}, expected "
            f"{settings.expected_embedding_dimension}"
        )

    client = qdrant_client()
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    else:
        info = client.get_collection(collection)
        existing_size = info.config.params.vectors.size
        if existing_size != vector_size:
            raise RuntimeError(
                f"Qdrant collection '{collection}' uses {existing_size}-dim vectors; "
                f"the configured embedding model returns {vector_size} dimensions"
            )

    points = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        page_number = chunk_rows[index][1]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"research-copilot:{version_id}:{index}"))
        points.append(models.PointStruct(
            id=point_id,
            vector=list(vector),
            payload={
                "paper_id": paper_id,
                "version_id": version_id,
                "chunk_index": index,
                "title": title,
                "source_filename": source_filename,
                "parser": parser_name,
                "page": page_number,
                "locator": f"page:{page_number}" if page_number else "document",
                "text": chunk,
            },
        ))

    # Reindexing must replace this version's points; otherwise a shorter new
    # parse leaves stale chunks in the collection and retrieval returns both.
    if replace_existing:
        client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(
                        key="version_id",
                        match=models.MatchValue(value=version_id),
                    )
                ])
            ),
            wait=True,
        )
    for offset in range(0, len(points), 256):
        client.upsert(collection_name=collection, points=points[offset : offset + 256], wait=True)
    return {"status": "indexed", "collection": collection, "chunks": len(points), "dimension": vector_size}
