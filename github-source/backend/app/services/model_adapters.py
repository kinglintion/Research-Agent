from __future__ import annotations

import json
import time
from functools import lru_cache

from openai import OpenAI

from ..config import get_settings


@lru_cache
def get_llm_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
    )


@lru_cache
def get_embedder():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=get_settings().embed_model, embed_batch_size=32)


@lru_cache
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model)


def check_llm(prompt: str) -> dict:
    settings = get_settings()
    started = time.perf_counter()
    response = get_llm_client().chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": "You are checking an OpenAI-compatible chat API. Follow the requested format exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=128,
    )
    content = response.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
        json_ok = isinstance(parsed, dict)
    except json.JSONDecodeError:
        json_ok = False
    return {
        "status": "ok",
        "model": response.model or settings.llm_model,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "details": {"json_output_ok": json_ok, "sample": content[:300]},
    }


def check_embedding(text: str) -> dict:
    settings = get_settings()
    started = time.perf_counter()
    vector = get_embedder().get_text_embedding(text)
    dimension = len(vector)
    expected = settings.expected_embedding_dimension
    return {
        "status": "ok" if dimension == expected else "dimension_mismatch",
        "model": settings.embed_model,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "details": {"dimension": dimension, "expected_dimension": expected},
    }


def check_reranker(query: str, passages: list[str]) -> dict:
    settings = get_settings()
    started = time.perf_counter()
    pairs = [(query, passage) for passage in passages]
    scores = get_reranker().predict(pairs)
    ranked = sorted(
        [
            {"index": i, "score": float(score), "passage": passages[i]}
            for i, score in enumerate(scores)
        ],
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "status": "ok",
        "model": settings.reranker_model,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "details": {"ranked_passages": ranked},
    }
