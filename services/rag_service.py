"""Protocol retrieval for Hospital Hybrid AI.

Default mode is local TF-IDF. If scikit-learn is unavailable, a simple local
keyword-overlap retriever is used so the clinical workflow does not fail.
OpenAI embeddings are optional and never required for the hospital demo.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "data" / "clinical_protocols.json"


def _docs():
    try:
        return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _doc_text(doc):
    return f"{doc.get('title','')} {' '.join(doc.get('topics', []))} {doc.get('content','')}"


def _tokens(text):
    return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2}


def _keyword_overlap(query: str, top_k: int):
    q = _tokens(query)
    scored = []
    for doc in _docs():
        d = _tokens(_doc_text(doc))
        union = len(q | d) or 1
        score = len(q & d) / union
        scored.append(({**doc, "retrieval_score": round(score, 4), "retrieval_mode": "keyword_fallback"}, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in scored[:top_k]]


def _tfidf(query: str, top_k: int):
    docs = _docs()
    if not docs:
        return []
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return _keyword_overlap(query, top_k)

    corpus = [_doc_text(doc) for doc in docs]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(corpus + [query])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    order = np.argsort(sims)[::-1][:top_k]
    return [
        {**docs[int(i)], "retrieval_score": round(float(sims[i]), 4), "retrieval_mode": "tfidf"}
        for i in order
    ]


def _openai_embeddings(query: str, top_k: int):
    try:
        import numpy as np
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("OpenAI embeddings dependencies are unavailable") from exc

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    docs = _docs()
    texts = [_doc_text(doc) for doc in docs]
    client = OpenAI(api_key=key)
    response = client.embeddings.create(model=model, input=[query] + texts)
    vectors = [np.asarray(item.embedding, dtype=float) for item in response.data]
    query_vector = vectors[0]
    query_norm = np.linalg.norm(query_vector) or 1.0
    scored = []
    for doc, vector in zip(docs, vectors[1:]):
        sim = float(np.dot(query_vector, vector) / (query_norm * (np.linalg.norm(vector) or 1.0)))
        scored.append(({**doc, "retrieval_score": round(sim, 4), "retrieval_mode": "openai_embeddings"}, sim))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in scored[:top_k]]


def retrieve_protocols(query: str, top_k: int = 3):
    mode = os.getenv("CAREAI_RAG_MODE", "tfidf").strip().lower()
    if mode == "openai_embeddings":
        try:
            return _openai_embeddings(query, top_k)
        except Exception:
            return _tfidf(query, top_k)
    return _tfidf(query, top_k)
