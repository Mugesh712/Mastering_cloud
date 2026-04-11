"""
RAG Retriever Module — PneumoCloud AI
========================================
Queries the Pinecone vector database for relevant medical context
based on a diagnosis result from DenseNet-121.

Used by:
  - gcp/main.py (Cloud Function — production)
  - Can also be imported locally for testing

Requires:
  - PINECONE_API_KEY environment variable
  - Pinecone index 'pneumocloud-medical-kb' populated by ingest.py
"""

import os

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
INDEX_NAME = "pneumocloud-medical-kb"
EMBEDDING_MODEL = "multilingual-e5-large"
TOP_K = 5  # Number of relevant chunks to retrieve


# ─────────────────────────────────────────────────────
# Lazy-initialized Pinecone client (avoids import cost
# if RAG is disabled or API key not set)
# ─────────────────────────────────────────────────────
_pc = None
_index = None


def _init_pinecone():
    """Initialize Pinecone client and index (lazy, once per container)."""
    global _pc, _index

    if _pc is not None:
        return True

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("[RAG] ⚠️ PINECONE_API_KEY not set — RAG disabled, using fallback.")
        return False

    try:
        from pinecone import Pinecone
        _pc = Pinecone(api_key=api_key)
        _index = _pc.Index(INDEX_NAME)
        print(f"[RAG] ✅ Connected to Pinecone index '{INDEX_NAME}'")
        return True
    except Exception as e:
        print(f"[RAG] ⚠️ Failed to connect to Pinecone: {e}")
        _pc = None
        _index = None
        return False


def build_query(disease: str, confidence: float,
                triage_level: str, department: str) -> str:
    """
    Build a semantic search query from the diagnosis result.

    The query is designed to retrieve the most relevant medical
    knowledge for generating a clinical report.
    """
    # Disease-specific query terms
    disease_queries = {
        "COVID": (
            "COVID-19 SARS-CoV-2 bilateral ground-glass opacities "
            "treatment protocol Remdesivir antiviral therapy isolation "
            "oxygen monitoring SpO2 follow-up post-COVID recovery "
            "dietary guidelines vitamin D zinc supplements drug interactions"
        ),
        "Viral Pneumonia": (
            "viral pneumonia interstitial infiltrates bilateral opacities "
            "antiviral therapy Oseltamivir supportive care oxygen therapy "
            "CURB-65 severity assessment antibiotic prophylaxis "
            "recovery nutrition immune boosting follow-up schedule"
        ),
        "Lung_Opacity": (
            "lung opacity differential diagnosis consolidation atelectasis "
            "pleural effusion pulmonary assessment CT imaging workup "
            "treatment protocol antibiotics follow-up monitoring "
            "dietary anti-inflammatory nutrition respiratory rehabilitation"
        ),
        "Normal": (
            "normal chest X-ray findings preventive care lung health "
            "annual screening lifestyle recommendations exercise "
            "balanced nutrition respiratory wellness vaccination schedule "
            "follow-up protocol healthy habits"
        ),
    }

    base_query = disease_queries.get(disease, f"{disease} chest X-ray treatment protocol clinical management")

    # Add severity context
    if triage_level in ("CRITICAL", "URGENT"):
        base_query += " emergency management ICU critical care monitoring"
    else:
        base_query += " outpatient management routine follow-up care"

    return base_query


def retrieve_medical_context(disease: str, confidence: float,
                              triage_level: str, department: str) -> str:
    """
    Main retrieval function — called by GCP Cloud Function.

    Args:
        disease:      Predicted disease class (e.g., 'COVID')
        confidence:   Model confidence (0.0-1.0)
        triage_level: Triage level (CRITICAL/URGENT/STANDARD/LOW)
        department:   Assigned department

    Returns:
        Formatted string of retrieved medical context with citations,
        or empty string if RAG is unavailable.
    """
    # Initialize Pinecone (lazy)
    if not _init_pinecone():
        return ""

    try:
        # Build search query
        query = build_query(disease, confidence, triage_level, department)
        print(f"[RAG] 🔍 Query: {query[:100]}...")

        # Embed query using Pinecone Inference API
        query_embedding = _pc.inference.embed(
            model=EMBEDDING_MODEL,
            inputs=[query],
            parameters={"input_type": "query"}
        )

        # Search the index
        results = _index.query(
            vector=query_embedding[0].values,
            top_k=TOP_K,
            include_metadata=True
        )

        if not results.matches:
            print("[RAG] ⚠️ No matching documents found.")
            return ""

        # Format retrieved context with citations
        context_parts = []
        seen_sections = set()

        for i, match in enumerate(results.matches, 1):
            metadata = match.metadata
            text = metadata.get("text", "")
            source = metadata.get("source_citation", "Unknown")
            section = metadata.get("section", "General")
            score = match.score

            # Deduplicate by section
            section_key = f"{metadata.get('source_file', '')}_{section}"
            if section_key in seen_sections:
                continue
            seen_sections.add(section_key)

            context_parts.append(
                f"[Reference {i} | Relevance: {score:.0%} | Source: {source}]\n{text}"
            )

        medical_context = "\n\n---\n\n".join(context_parts)
        print(f"[RAG] ✅ Retrieved {len(context_parts)} relevant medical references (top score: {results.matches[0].score:.2%})")

        return medical_context

    except Exception as e:
        print(f"[RAG] ⚠️ Retrieval failed: {e}")
        return ""


# ─────────────────────────────────────────────────────
# CLI test mode
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PneumoCloud AI — RAG Retriever Test")
    print("=" * 60)

    test_cases = [
        ("COVID", 0.91, "CRITICAL", "Pulmonology — Isolation Ward"),
        ("Viral Pneumonia", 0.88, "URGENT", "Pulmonology — ICU"),
        ("Lung_Opacity", 0.85, "URGENT", "Pulmonology"),
        ("Normal", 0.95, "LOW", "General Outpatient"),
    ]

    for disease, conf, triage, dept in test_cases:
        print(f"\n{'─' * 60}")
        print(f"Testing: {disease} (conf={conf}, triage={triage})")
        print(f"{'─' * 60}")

        context = retrieve_medical_context(disease, conf, triage, dept)

        if context:
            # Show first 500 chars
            print(f"\n📄 Retrieved context ({len(context)} chars):")
            print(context[:500])
            print("..." if len(context) > 500 else "")
        else:
            print("⚠️ No context retrieved (check API key and index)")

    print(f"\n{'=' * 60}")
    print("  Test complete!")
    print(f"{'=' * 60}")
