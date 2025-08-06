from pydantic import BaseModel


class Answer(BaseModel):
    choice: int
    reason: str