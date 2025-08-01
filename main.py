from fastapi import FastAPI
from pydantic import BaseModel
from rag_index import load_rag_index, load_llm, get_vector_store, load_embed
from llama_index.core.query_engine import RetrieverQueryEngine
from rag_retriever import get_retriever
from ultils import get_doc_ids
app = FastAPI()

# Load index khi start server
# Model embedding
embed_model = load_embed()
vector_store = get_vector_store("ITHelpDesk")
llm = load_llm()
retriever = get_retriever(vector_store=vector_store, embed_model=embed_model, query_mode="default", similarity_top_k = 2)
query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

class QuestionRequest(BaseModel):
    question: str

class ChatMessageResponse(BaseModel):
    answer: str
    sourceDocuments: list[int]


@app.post("/ask", response_model=ChatMessageResponse)
async def ask_question(req: QuestionRequest):
    response = query_engine.query(req.question)
    source_docs = get_doc_ids(response.source_nodes)

    return ChatMessageResponse(
        answer=str(response),
        sourceDocuments=source_docs
    )
