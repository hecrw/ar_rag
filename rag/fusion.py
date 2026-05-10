"""Result list fusion (Reciprocal Rank Fusion)."""


def rrf_fuse(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Combine multiple ranked lists into one via Reciprocal Rank Fusion.

    Each input list is a list of dicts. Items are matched by their "id" field.
    """
    scores: dict = {}
    canonical: dict = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.get("id") or id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in canonical:
                canonical[doc_id] = doc
    fused_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [{**canonical[did], "score": scores[did]} for did in fused_ids]
