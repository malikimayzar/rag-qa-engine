import os
from groq import Groq
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()
logger = get_logger("groq_llm")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """Kamu adalah asisten QA yang menjawab pertanyaan berdasarkan konteks dokumen yang diberikan.
Aturan:
1. Jawab HANYA berdasarkan konteks yang diberikan
2. Jika jawaban tidak ada di konteks, katakan "Informasi tidak ditemukan dalam dokumen"
3. Selalu sertakan sumber (doc_id + halaman) di akhir jawaban
4. Jawab dalam bahasa yang sama dengan pertanyaan (Indo/Inggris)
5. Jawaban harus ringkas dan faktual"""

def format_context(chunks: list[dict]) -> str:
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[{i}] Halaman {chunk['page']}:\n{chunk['text']}\n"
    return context

def format_sources(chunks: list[dict]) -> str:
    sources = []
    seen = set()
    for chunk in chunks:
        key = f"{chunk['doc_id']}-{chunk['page']}"
        if key not in seen:
            seen.add(key)
            sources.append(f"- {chunk['doc_id']}, halaman {chunk['page']}")
    return "\n".join(sources)

def answer(query: str, chunks: list[dict]) -> dict:
    context = format_context(chunks)
    sources = format_sources(chunks)
    prompt = f"""Konteks dokumen:
{context}
Pertanyaan: {query}
Jawab berdasarkan konteks di atas."""
    logger.info(f"Sending to Groq: {query[:50]}...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.1
    )
    answer_text = response.choices[0].message.content
    usage = response.usage
    logger.info(f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")
    return {
        "query": query,
        "answer": answer_text,
        "sources": sources,
        "chunks_used": len(chunks),
        "tokens_in": usage.prompt_tokens,
        "tokens_out": usage.completion_tokens
    }

def answer_stream(query: str, chunks: list[dict]):
    """Generator that yields token strings, then sources at the end."""
    context = format_context(chunks)
    sources = format_sources(chunks)
    prompt = f"""Konteks dokumen:
{context}
Pertanyaan: {query}
Jawab berdasarkan konteks di atas."""
    logger.info(f"Streaming to Groq: {query[:50]}...")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.1,
        stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
    # Send sources as last event
    yield f"__SOURCES__:{sources}"