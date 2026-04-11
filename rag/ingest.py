"""
RAG Ingestion Script — PneumoCloud AI
========================================
One-time script to:
  1. Read all medical knowledge documents from knowledge_base/
  2. Split them into ~400-token chunks with overlap
  3. Embed each chunk using Pinecone Inference API
  4. Upsert all vectors into a Pinecone serverless index

Usage:
  export PINECONE_API_KEY="your-api-key"
  python rag/ingest.py
"""

import os
import re
import time
import hashlib
from pinecone import Pinecone, ServerlessSpec

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
INDEX_NAME = "pneumocloud-medical-kb"
EMBEDDING_MODEL = "multilingual-e5-large"
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHUNK_SIZE = 1600        # ~400 tokens (4 chars ≈ 1 token)
CHUNK_OVERLAP = 200      # overlap between chunks for context continuity
BATCH_SIZE = 50          # vectors per upsert batch


def get_api_key():
    """Get Pinecone API key from environment."""
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set.")
        print("   Run: export PINECONE_API_KEY='your-api-key'")
        raise SystemExit(1)
    return api_key


def chunk_document(text: str, source_file: str) -> list[dict]:
    """
    Split a document into chunks, preserving section headers as context.

    Strategy:
      1. Split on markdown headers (## or ###)
      2. If a section is too long, split it into overlapping chunks
      3. Each chunk carries metadata (source, section, chunk_id)
    """
    chunks = []

    # Split on markdown headers
    sections = re.split(r'\n(?=###?\s)', text)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 50:
            continue

        # Extract section header if present
        header_match = re.match(r'^(###?\s+.+)', section)
        header = header_match.group(1).strip() if header_match else "General"

        # Extract source citation if present
        source_match = re.findall(r'\[Source:\s*(.+?)\]', section)
        source_citation = source_match[-1] if source_match else source_file

        # If section fits in one chunk, keep it whole
        if len(section) <= CHUNK_SIZE:
            chunk_id = hashlib.md5(section[:100].encode()).hexdigest()[:12]
            chunks.append({
                "id": f"{source_file}_{chunk_id}",
                "text": section,
                "metadata": {
                    "source_file": source_file,
                    "section": header,
                    "source_citation": source_citation,
                    "char_count": len(section),
                }
            })
        else:
            # Split long sections into overlapping chunks
            words = section.split()
            total_chars = 0
            current_chunk_words = []

            for word in words:
                current_chunk_words.append(word)
                total_chars += len(word) + 1

                if total_chars >= CHUNK_SIZE:
                    chunk_text = " ".join(current_chunk_words)
                    chunk_id = hashlib.md5(chunk_text[:100].encode()).hexdigest()[:12]
                    chunks.append({
                        "id": f"{source_file}_{chunk_id}",
                        "text": chunk_text,
                        "metadata": {
                            "source_file": source_file,
                            "section": header,
                            "source_citation": source_citation,
                            "char_count": len(chunk_text),
                        }
                    })

                    # Keep overlap
                    overlap_words = current_chunk_words[-CHUNK_OVERLAP // 5:]
                    current_chunk_words = list(overlap_words)
                    total_chars = sum(len(w) + 1 for w in current_chunk_words)

            # Remaining words
            if current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                if len(chunk_text) > 50:  # Skip tiny remainders
                    chunk_id = hashlib.md5(chunk_text[:100].encode()).hexdigest()[:12]
                    chunks.append({
                        "id": f"{source_file}_{chunk_id}",
                        "text": chunk_text,
                        "metadata": {
                            "source_file": source_file,
                            "section": header,
                            "source_citation": source_citation,
                            "char_count": len(chunk_text),
                        }
                    })

    return chunks


def load_all_documents() -> list[dict]:
    """Load and chunk all markdown documents from knowledge_base/."""
    all_chunks = []

    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"❌ Knowledge directory not found: {KNOWLEDGE_DIR}")
        raise SystemExit(1)

    md_files = sorted([f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith(".md")])
    print(f"📂 Found {len(md_files)} knowledge documents:\n")

    for filename in md_files:
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_document(text, filename.replace(".md", ""))
        all_chunks.extend(chunks)
        print(f"   ✅ {filename:<40} → {len(chunks):>3} chunks ({len(text):,} chars)")

    print(f"\n📊 Total chunks: {len(all_chunks)}")
    return all_chunks


def create_index(pc: Pinecone):
    """Create the Pinecone serverless index if it doesn't exist."""
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME in existing:
        print(f"📌 Index '{INDEX_NAME}' already exists. Will upsert into it.")
        return

    print(f"🔨 Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,  # multilingual-e5-large output dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    # Wait for index to be ready
    print("   ⏳ Waiting for index to initialize...")
    while not pc.describe_index(INDEX_NAME).status.get("ready", False):
        time.sleep(2)
    print("   ✅ Index is ready!")


def embed_and_upsert(pc: Pinecone, chunks: list[dict]):
    """Embed all chunks using Pinecone Inference and upsert to the index."""
    index = pc.Index(INDEX_NAME)

    total = len(chunks)
    upserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]

        # Embed using Pinecone Inference API
        embeddings = pc.inference.embed(
            model=EMBEDDING_MODEL,
            inputs=texts,
            parameters={"input_type": "passage"}
        )

        # Build upsert vectors
        vectors = []
        for j, chunk in enumerate(batch):
            vectors.append({
                "id": chunk["id"],
                "values": embeddings[j].values,
                "metadata": {
                    **chunk["metadata"],
                    "text": chunk["text"],  # Store text in metadata for retrieval
                }
            })

        # Upsert batch
        index.upsert(vectors=vectors)
        upserted += len(batch)
        print(f"   📤 Upserted {upserted}/{total} vectors...")

    print(f"\n✅ All {total} vectors upserted to '{INDEX_NAME}'!")

    # Show index stats
    time.sleep(2)
    stats = index.describe_index_stats()
    print(f"📊 Index stats: {stats.total_vector_count} total vectors")


def main():
    print("=" * 60)
    print("  PneumoCloud AI — RAG Knowledge Base Ingestion")
    print("=" * 60)
    print()

    # 1. Init Pinecone
    api_key = get_api_key()
    pc = Pinecone(api_key=api_key)
    print("✅ Pinecone client initialized\n")

    # 2. Load and chunk documents
    chunks = load_all_documents()
    print()

    # 3. Create index
    create_index(pc)
    print()

    # 4. Embed and upsert
    print("🔄 Embedding and upserting chunks...")
    embed_and_upsert(pc, chunks)

    print()
    print("=" * 60)
    print("  ✅ Ingestion complete!")
    print(f"  Index: {INDEX_NAME}")
    print(f"  Chunks: {len(chunks)}")
    print("  Ready for RAG queries from GCP Cloud Function")
    print("=" * 60)


if __name__ == "__main__":
    main()
