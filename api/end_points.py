from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from config.embed import load_embed
from config.llm import load_llm
from config.vector_store import get_vector_store
from router.router import routing

router = APIRouter()

# Load models and vector store once
embed_model = load_embed()
llm = load_llm()
vector_store = get_vector_store("Departments")


class QuestionRequest(BaseModel):
    question: str
    department_id: int


class ChatMessageResponse(BaseModel):
    answer: str
    sourceDocuments: List[int] | None


@router.post("/ask", response_model=ChatMessageResponse)
async def ask_question(req: QuestionRequest):
    response_text, source_docs = routing(
        query_str=req.question,
        user_department_id=req.department_id,
        vector_store=vector_store,
        embed_model=embed_model,
        llm=llm,
    )

    return ChatMessageResponse(
        answer=response_text,
        sourceDocuments=source_docs,
    )