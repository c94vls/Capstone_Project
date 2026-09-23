"""
support_assistant/main.py
FastAPI application serving the POST /ask endpoint.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from graph import support_graph, SupportResponse
import ingest

app = FastAPI(
    title="Zepto Customer Support Assistant",
    description="Grounded GenAI Policy Assistant backed by ChromaDB and LangGraph",
    version="1.0.0"
)

class AskRequest(BaseModel):
    query: str = Field(..., example="What is Zepto's return policy for perishable items?")

@app.on_event("startup")
def startup_event():
    # Automatically build/verify vector index on launch
    ingest.build_vector_store()

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "zepto-support-assistant"}

@app.post("/ask", response_model=SupportResponse)
def ask_support(request: AskRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    initial_state = {
        "query": request.query,
        "intent": "",
        "retrieved_chunks": [],
        "response": {},
        "retry_count": 0
    }

    final_state = support_graph.invoke(initial_state)
    return final_state["response"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)