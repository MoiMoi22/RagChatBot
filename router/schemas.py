from pydantic import BaseModel


class AnswerRouter(BaseModel):
    choice: int
    reason: str

class AnswerQuery(BaseModel):
    query: str