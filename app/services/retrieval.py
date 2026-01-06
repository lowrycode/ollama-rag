def distance_to_relevance(distance: float, max_distance: float = 1.0) -> float:
    """
    Converts cosine distance to a user-facing relevance score.
    1.0 = very relevant
    0.0 = not relevant
    """
    relevance = 1.0 - min(distance / max_distance, 1.0)
    return round(relevance, 3)


def retrieve_context(
    vector_db,
    query: str,
    k: int = 12,
    max_distance: float = 0.9,  # filter out irrelevant info
    top_k: int = 4,  # how many to return
):
    retrieved = vector_db.similarity_search_with_score(query, k=k)

    filtered = []
    for doc, distance in retrieved:
        if distance > max_distance:
            continue

        filtered.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content,
                "distance": distance,  # internal / debug only
                "relevance": distance_to_relevance(distance),
            }
        )

    # Best → worst (lowest distance first)
    filtered.sort(key=lambda x: x["distance"])

    return filtered[:top_k]


def generate_prompt_string(context: str, question: str) -> str:
    return f"""You are a helpful assistant.

Answer the question clearly and directly for an end user.
Write as if the information is part of your own knowledge.

Use only the information provided below.
If the answer is not contained in the information, say:
"I don't have enough information to answer that question."

Information:
{context}

Question:
{question}

Style rules:
- Do not mention sources, documents, or context
- Do not explain how you know the answer
- Answer in plain, direct language

Answer:
"""
