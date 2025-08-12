import json
from typing import Type, TypeVar, Generic, List
from llama_index.core.types import BaseOutputParser
from utils.utils import _marshal_output_to_json, _escape_curly_braces


T = TypeVar("T")  # kiểu tổng quát cho model (ví dụ Answer)

class CustomOutputParser(Generic[T], BaseOutputParser):
    def __init__(self, model_type: Type[T]):
        """
        model_type: lớp của đối tượng đích để parse (ví dụ Answer).
        Lớp này nên có phương thức `model_validate` (Pydantic v2) hoặc tương đương.
        """
        self.model_type = model_type

    def parse(self, output: str) -> List[T]:
        """Parse string thành list[T]."""
        json_output = _marshal_output_to_json(output)
        json_dicts = json.loads(json_output)
        return [self.model_type.model_validate(d) for d in json_dicts]

    def format(self, prompt_template: str, format_str: str) -> str:
        """Ghép prompt + format_str do người dùng cung cấp."""
        return prompt_template + "\n\n" + _escape_curly_braces(format_str)