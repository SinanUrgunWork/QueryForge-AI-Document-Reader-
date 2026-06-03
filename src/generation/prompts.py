from langchain.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise assistant. Answer the question using ONLY
the context below. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
)
