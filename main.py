from fastapi import FastAPI
from pydantic import BaseModel
from rag_index import load_rag_index, load_llm

app = FastAPI()

# Load index khi start server
index = load_rag_index()
llm = load_llm()
query_engine = index.as_query_engine(llm)

class QuestionRequest(BaseModel):
    question: str

class ChatMessageResponse(BaseModel):
    answer: str
    sourceDocuments: list[str]

@app.post("/ask", response_model=ChatMessageResponse)
async def ask_question(req: QuestionRequest):
    response = query_engine.query(req.question)
    response = "Hello"
    # Phát triển thêm
    source_docs = ['123']
    # if hasattr(response, "source_nodes"):
    #     source_docs = [node.node.text for node in response.source_nodes]

    return ChatMessageResponse(
        answer=str(response),
        sourceDocuments=source_docs
    )
