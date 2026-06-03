from langchain_openai import ChatOpenAI
from src.generation.prompts import RAG_PROMPT
from src.retrieval.store import search
from src.config import LLM_MODEL, TEMPERATURE, MAX_TOKENS, TOP_K


def run(question: str, top_k: int = TOP_K) -> dict:
    chunks = search(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "chunks_used": 0,
        }

    context = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_id']}]\n{c['content']}"
        for c in chunks
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    prompt = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": chunks,
        "chunks_used": len(chunks),
    }
