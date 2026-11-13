import streamlit as st
import ollama

# Page configuration
st.set_page_config(page_title="Local RAG Engine", page_icon="🤖", layout="centered")

st.title("🤖 Local RAG Support Bot")
st.caption("Powered by Llama 3.2 (3B) running locally via Ollama on RTX 3050")

# 1. Mock database retrieval function
def mock_fetch_database_chunks(query: str, client_id: str):
    return [
        {
            "content": "UPI dispute turnaround time for merchant chargebacks is strictly 48 hours.",
            "similarity_score": 0.82,  # Set below 0.65 to test refusal logic
            "doc_name": "PhonePe_UPI_SOP.pdf",
            "doc_version": "v2.0",
            "page_number": 4
        }
    ]

# 2. Core Local RAG Pipeline
def process_rag_query(query_text: str, client_id: str):
    retrieved_chunks = mock_fetch_database_chunks(query_text, client_id)
    top_score = retrieved_chunks[0]["similarity_score"] if retrieved_chunks else 0.0

    # Anti-hallucination cosine similarity gate
    if top_score < 0.65:
        return {
            "status": "refused",
            "answer": "Information not found in internal context documentation.",
            "citations": []
        }

    context_text = "\n".join([f"- {c['content']}" for c in retrieved_chunks])
    system_prompt = f"""You are an internal support bot. Answer using ONLY the provided document context below.

Context:
{context_text}

Question: {query_text}
Answer:"""

    response = ollama.chat(
        model='llama3.2:3b',
        options={'temperature': 0.0},
        messages=[{'role': 'user', 'content': system_prompt}]
    )

    citations = [
        {"doc_name": c["doc_name"], "version": c["doc_version"], "page_number": c["page_number"]}
        for c in retrieved_chunks
    ]

    return {
        "status": "success",
        "answer": response['message']['content'].strip(),
        "citations": citations
    }

# UI Input Components
client_id = st.text_input("Client ID", value="PhonePe")
user_query = st.text_area("Enter your query:", value="What is the turnaround time for merchant chargebacks?")

if st.button("Run Query", type="primary"):
    with st.spinner("Querying local Llama 3.2 model on GPU..."):
        result = process_rag_query(user_query, client_id)
        
    st.markdown("### Answer")
    if result["status"] == "refused":
        st.warning(result["answer"])
    else:
        st.success(result["answer"])
        
        st.markdown("### Citations")
        for cite in result["citations"]:
            st.info(f"📄 **Document:** {cite['doc_name']} (Version {cite['version']}) — **Page {cite['page_number']}**")