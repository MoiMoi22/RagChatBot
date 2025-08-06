from llama_index.core import PromptTemplate


choices = [
    "DEPARTMENT_QUERY: Loại câu hỏi này liên quan đến một phòng ban cụ thể trong công ty. Thường đề cập đến chính sách, quy trình, nhân sự hoặc công việc của các phòng như Nhân sự, Kỹ thuật, Kinh doanh, Tài chính... Trả lời yêu cầu truy xuất tài liệu nội bộ của phòng ban tương ứng",
    "CHITCHAT_OR_GENERAL: Loại câu hỏi này mang tính trò chuyện (chitchat) hoặc kiến thức chung, không liên quan đến bất kỳ phòng ban cụ thể nào. Bao gồm lời chào hỏi, câu hỏi về AI, kiến thức xã hội, giải trí hoặc các câu hỏi mở",
]

router_prompt0 = PromptTemplate(
    "Bạn là một chuyên gia phân loại truy vấn. Dưới đây là một câu hỏi từ người dùng và hai lựa chọn xử lý có thể có. Hãy chọn ra lựa chọn phù hợp nhất để xử lý câu hỏi." 
    ". Được đưa ra dưới dạng danh sách có số thứ tự (1 đến"
    " {num_choices}), mỗi lựa chọn trong danh sách tương đương với 1 summary"
    " .\n---------------------\n{context_list}\n---------------------\n"
    " Chỉ sử dụng các lựa chọn ở trên và không sử kiến thức ở ngoài, trả về những lựa chọn tốt nhất"
    " (không lớn hơn {max_outputs}, nhưng chỉ lấy những lựa chọn cần thiết)"
    " Đó là những lựa chọn liên quan nhất đến câu hỏi: '{query_str}'\n"
)

FORMAT_STR = """Output nên được format như là 1 JSON instance mà theo 
JSON schema ở dưới. 

Đây là đầu ra của schema:
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "choice": {
        "type": "integer"
      },
      "reason": {
        "type": "string"
      }
    },
    "required": [
      "choice",
      "reason"
    ],
    "additionalProperties": false
  }
}
"""