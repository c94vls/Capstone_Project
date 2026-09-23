# Support Assistant Module (`/support_assistant`)

## 1. RAG Pipeline Architecture Walkthrough

[Customer Query]
│
▼
+─────────────────────────────────────────────────────────────+
| LangGraph Node: classify_intent                             |
| Mode MOCK_LLM=1: Keyword heuristic (delivery, return, etc.) |
+─────────────────────────────────────────────────────────────+
│
├──[ intent == "policy_question" ]──► +──────────────────────────────────────────────+
│                                     | LangGraph Node: retrieve_and_answer          |
│                                     | 1. Embed query (all-MiniLM-L6-v2)             |
│                                     | 2. Vector search: ChromaDB top-3 (cosine)    |
│                                     | 3. Format grounded Pydantic response         |
│                                     +──────────────────────────────────────────────+
│                                                              │
└──[ intent == "general_question" ]─► +──────────────────────────────────────────────+
| LangGraph Node: direct_answer                |
| Outputs canned refusal & empty sources       |
+──────────────────────────────────────────────+
│
▼
[Validated JSON Output]
(answer, sources, confidence)


### Stage-by-Stage Component Responsibilities
1. **Ingestion (`ingest.py`):** Reads the 8 canonical Zepto policy files (`docs/doc_01.txt` through `doc_08.txt`), splits into document-level chunks, and embeds them using `sentence-transformers/all-MiniLM-L6-v2`.
2. **Vector Storage:** Persisted locally in ChromaDB under collection `zepto_policies` using cosine similarity metrics.
3. **Intent Routing (`graph.py: classify_intent`):**
   * **`MOCK_LLM=1` (Graded Baseline):** Evaluates whether the incoming query contains domain keywords (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`). If matched $\rightarrow$ routes to `retrieve_and_answer`; otherwise $\rightarrow$ routes to `direct_answer`.
   * **`MOCK_LLM=0` (Optional):** Prompts an external LLM to perform intent categorization.
4. **Retrieval (`graph.py: retrieve_and_answer`):** Performs local vector search against ChromaDB, extracting the top 3 most similar document passages.
5. **Generation & Schema Enforcement:**
   * **`MOCK_LLM=1`:** Generates deterministically: `f"Based on the retrieved context: {top_chunk_snippet}"`, extracting the citation ID directly from Chroma metadata and fixing confidence to `1.0`.
   * **`MOCK_LLM=0`:** Injects retrieved context into `STRUCTURED_PROMPT_TEMPLATE`. Parses output against `SupportResponse` Pydantic model with up to 2 retry attempts.

---

## 2. Local Setup & Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build local vector database
python support_assistant/ingest.py

# 3. Start FastAPI server locally
uvicorn support_assistant.main:app --host 127.0.0.1 --port 7860