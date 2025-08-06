from typing import List
from router.schemas import Answer
from llama_index.core.base.response.schema import Response
import re


def get_choice_str(choices):
    choices_str = "\n\n".join(
        [f"{idx+1}. {c}" for idx, c in enumerate(choices)]
    )
    return choices_str

def _marshal_output_to_json(output: str) -> str:
    output = output.strip()
    left = output.find("[")
    right = output.find("]")
    output = output[left : right + 1]
    return output

def _escape_curly_braces(input_string: str) -> str:
    # Replace '{' with '{{' and '}' with '}}' to escape curly braces
    escaped_string = input_string.replace("{", "{{").replace("}", "}}")
    return escaped_string

def extract_choices(answers: List[Answer]) -> List[int]:
    return [answer.choice for answer in answers]

def extract_answer(response: Response):
    doc_ids = response.metadata.get("doc_ids", None)

    if doc_ids == None:
        return remove_think_tags(response.response), None
    else:
        return  remove_think_tags('RAG: ' + response.response.text), doc_ids
    
def remove_think_tags(text):
    # Kiểm tra có tồn tại <think>...</think> không
    if "<think>" in text and "</think>" in text:
        # Xóa toàn bộ đoạn từ <think> đến </think> (bao gồm luôn cả hai thẻ)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text