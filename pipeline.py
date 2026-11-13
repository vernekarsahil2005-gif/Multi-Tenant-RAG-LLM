import os
import fitz  # PyMuPDF
import docx
import spacy
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================
# 1. LOAD LOCAL MODELS
# ==========================================
print("[INIT] Loading spaCy NLP engine...")
nlp = spacy.load("en_core_web_sm")

print("[INIT] Loading Bi-Encoder Embedding Model...")
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("[INIT] Loading Cross-Encoder Re-ranker...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Competitor registry for client isolation guardrail
KNOWN_COMPETITORS = {
    "phonepe": ["google pay", "gpay", "paytm", "bharatpe", "cred"],
    "google pay": ["phonepe", "paytm", "bharatpe", "cred"],
    "paytm": ["phonepe", "google pay", "gpay", "bharatpe"],
    "bluedart": ["delhivery", "ecom express", "dtdc", "fedex"],
    "medanta": ["apollo", "fortis", "max healthcare"]
}

# ==========================================
# 2. DOCUMENT INGESTION & CHUNKING
# ==========================================
def extract_text(file_path: str) -> str:
    """Extract raw text from PDF, DOCX, or TXT."""
    ext = os.path.splitext(file_path)[-1].lower()
    text = ""
    
    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text() + "\n"
    elif ext == ".docx":
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]

# ==========================================
# 3. SPACY NER PRE-RETRIEVAL GUARDRAIL
# ==========================================
def check_entity_guardrail(query: str, active_client_id: str) -> bool:
    """
    Returns False if the query contains an unauthorized competitor entity.
    """
    doc = nlp(query)
    detected_orgs = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
    
    # Also perform a simple token search in case NER misses casual casing
    query_lower = query.lower()
    banned_entities = KNOWN_COMPETITORS.get(active_client_id.lower(), [])
    
    for competitor in banned_entities:
        if competitor in query_lower or any(competitor in org for org in detected_orgs):
            print(f"[SECURITY ALERT] Competitor entity '{competitor}' detected in session for client '{active_client_id}'.")
            return False
            
    return True

# ==========================================
# 4. RETRIEVAL, RE-RANKING & SAFETY GATE
# ==========================================
def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def process_query(query: str, client_id: str, knowledge_chunks: list, threshold: float = 0.4) -> str:
    """
    Main execution pipeline for answering queries.
    """
    # Step A: Pre-retrieval Security Check
    if not check_entity_guardrail(query, client_id):
        return (
            f"[Access Denied]: The query contains entities outside the authorized "
            f"knowledge scope for client '{client_id.upper()}'."
        )

    if not knowledge_chunks:
        return "[Error]: No active documents found for this client."

    # Step B: Bi-Encoder Dense Retrieval
    query_embedding = bi_encoder.encode(query)
    chunk_embeddings = bi_encoder.encode(knowledge_chunks)
    
    scored_chunks = []
    for idx, chunk in enumerate(knowledge_chunks):
        sim = cosine_similarity(query_embedding, chunk_embeddings[idx])
        scored_chunks.append((chunk, float(sim)))
        
    # Sort by initial similarity score descending
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored_chunks[:5]

    # Step C: Anti-Hallucination Gate (Check against threshold)
    best_bi_score = top_candidates[0][1]
    print(f"[METRIC] Top Bi-Encoder Cosine Score: {best_bi_score:.4f}")
    if best_bi_score < threshold:
        return (
            "[Safety Gate Triggered]: The query does not sufficiently match official "
            f"SOP documentation for {client_id.upper()} (Similarity: {best_bi_score:.2f} < {threshold}). "
            "Halting generation to prevent hallucination."
        )

    # Step D: Cross-Encoder Re-Ranking
    cross_pairs = [[query, item[0]] for item in top_candidates]
    cross_scores = cross_encoder.predict(cross_pairs)
    
    ranked_chunks = [top_candidates[i][0] for i in np.argsort(cross_scores)[::-1]]
    final_context = "\n\n".join(ranked_chunks[:2])

    # Step E: Ollama Local LLM Synthesis
    prompt = f"""You are an internal operations support AI for client: {client_id.upper()}.
Answer the question strictly using the provided context below. 
If the answer cannot be found in the context, state that it is not documented.

Context:
{final_context}

Question: {query}
Answer:"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    
    return response["message"]["content"]

# ==========================================
# 5. SELF-TEST RUNNER
# ==========================================
if __name__ == "__main__":
    print("\n--- RUNNING LOCAL PIPELINE INTEGRATION TEST ---")
    
    # Mock tenant documentation for testing
    sample_docs = [
        "Refund SOP PhonePe: For failed UPI transactions where money is debited, the auto-refund cycle is T+2 banking days.",
        "PhonePe Merchant Settlement: Settlement batches run at 06:00 AM, 02:00 PM, and 10:00 PM daily into the merchant's verified bank account.",
        "General IT Policy: Contact desktop support on extension 4402 for VPN password resets."
    ]
    
    # Test 1: Legitimate Query
    print("\n[TEST 1] Legitimate in-domain query:")
    res1 = process_query(
        query="What is the refund timeline for failed UPI transactions?",
        client_id="PhonePe",
        knowledge_chunks=sample_docs
    )
    print("Response:\n", res1)

    # Test 2: Competitor Boundary Violation (spaCy Guardrail)
    print("\n[TEST 2] Cross-tenant competitor query:")
    res2 = process_query(
        query="How does this compare to Google Pay refund timelines?",
        client_id="PhonePe",
        knowledge_chunks=sample_docs
    )
    print("Response:\n", res2)

    # Test 3: Out-of-Domain Query (0.65 Safety Gate)
    print("\n[TEST 3] Out-of-domain query (safety gate trigger):")
    res3 = process_query(
        query="What is the recipe for chocolate chip cookies?",
        client_id="PhonePe",
        knowledge_chunks=sample_docs
    )
    print("Response:\n", res3)