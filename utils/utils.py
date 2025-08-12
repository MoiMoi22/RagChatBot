from __future__ import annotations  # <- quan trọng, hoãn đánh giá annotation
from typing import List, TYPE_CHECKING
from llama_index.core.base.response.schema import Response
import re
from underthesea import word_tokenize
import stopwordsiso as stop

if TYPE_CHECKING:
    # chỉ dùng lúc type-check, KHÔNG chạy ở runtime -> không tạo phụ thuộc import
    from router.schemas import AnswerRouter, AnswerQuery

def get_choice_str(choices):
    return "\n\n".join(f"{idx+1}. {c}" for idx, c in enumerate(choices))

def _marshal_output_to_json(output: str) -> str:
    output = output.strip()
    left = output.find("[")
    right = output.find("]")
    return output[left : right + 1]

def _escape_curly_braces(input_string: str) -> str:
    return input_string.replace("{", "{{").replace("}", "}}")

def extract_choices(answers: List["AnswerRouter"]) -> List[int]:
    return [answer.choice for answer in answers]

def extract_queries(items: List["AnswerQuery"]) -> List[str]:
    return [item.query for item in items]

def extract_answer(response: Response):
    doc_ids = response.metadata.get("doc_ids", None)
    if doc_ids is None:
        return remove_think_tags(response.response), None
    else:
        return remove_think_tags('RAG: ' + response.response.text), doc_ids

def remove_think_tags(text):
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text

stopwords_vi = stop.stopwords("vi")

def vn_tokenizer_no_stopword(text: str):
    tokens = word_tokenize(text, format="list")
    return [t for t in tokens if t.lower() not in stopwords_vi]
