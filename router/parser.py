import json
from typing import List
from router.schemas import Answer
from router.prompt import FORMAT_STR
from llama_index.core.types import BaseOutputParser
from utils.utils import _marshal_output_to_json, _escape_curly_braces


class RouterOutputParser(BaseOutputParser):
    def parse(self, output: str) -> List[Answer]:
        """Parse string."""
        json_output = _marshal_output_to_json(output)
        json_dicts = json.loads(json_output)
        answers = [Answer.model_validate(json_dict) for json_dict in json_dicts]
        return answers

    def format(self, prompt_template: str) -> str:
        return prompt_template + "\n\n" + _escape_curly_braces(FORMAT_STR)