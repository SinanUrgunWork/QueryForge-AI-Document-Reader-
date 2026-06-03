def score_relevance(question: str, chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    scores = [c.get("score", 0.0) for c in chunks]
    return round(sum(scores) / len(scores), 3)


def score_faithfulness(answer: str, chunks: list[dict]) -> float:
    if not chunks or not answer:
        return 0.0
    context_words = set(
        " ".join(c["content"] for c in chunks).lower().split()
    )
    answer_words = answer.lower().split()
    if not answer_words:
        return 0.0
    overlap = sum(1 for w in answer_words if w in context_words)
    return round(overlap / len(answer_words), 3)
