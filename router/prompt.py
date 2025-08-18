from llama_index.core import PromptTemplate

#ROUTER
choices_router = [
    "DEPARTMENT_QUERY: Loại câu hỏi này liên quan đến một phòng ban cụ thể trong công ty. Thường đề cập đến chính sách, quy trình, nhân sự hoặc công việc của các phòng như Nhân sự, Kỹ thuật, Kinh doanh, Tài chính... Trả lời yêu cầu truy xuất tài liệu nội bộ của phòng ban tương ứng. Ngoài ra có phòng ban it helper, chuyên xử lý các vấn đề liên quan đến mạng hay các ứng dụng hay được dùng trong công ty, và có phòng ban xử lý liên quan đến ielts",
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

FORMAT_OUTPUT_ROUTER = """Output nên được format như là 1 JSON instance mà theo 
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

query_gen_prompt = PromptTemplate(
    "Bạn là một trợ lý hữu ích, có nhiệm vụ tạo ra nhiều câu truy vấn tìm kiếm tiếng Việt "
    "dựa trên một câu truy vấn đầu vào. Hãy tạo {num_queries} câu truy vấn tìm kiếm, "
    "mỗi câu trên một dòng, giữ nguyên ý nghĩa nhưng diễn đạt khác nhau, "
    "có nội dung liên quan chặt chẽ đến câu truy vấn đầu vào:\n"
    "Câu truy vấn: {query}\n"
    "Các câu truy vấn:\n"
)

FORMAT_OUTPUT_ANSWER_QUERIES = """Output nên được format như là 1 JSON instance mà theo 
JSON schema ở dưới. 

Đây là đầu ra của schema:
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string"
      }
    },
    "required": [
      "query"
    ],
    "additionalProperties": false
  }
}
"""


QUESTION_GEN_USER_TMPL = (
    "Thông tin ngữ cảnh được trình bày bên dưới.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Dựa trên thông tin ngữ cảnh và không dùng kiến thức có sẵn, "
    "hãy tạo ra các câu hỏi liên quan. "
)

QUESTION_GEN_SYS_TMPL = """
Bạn là một Giáo viên/ Giảng viên. Nhiệm vụ của bạn là tạo ra 
{num_questions_per_chunk} câu hỏi cho một bài kiểm tra/ kỳ thi sắp tới. 
Các câu hỏi cần đa dạng về tính chất trong toàn bộ tài liệu. 
Hãy giới hạn các câu hỏi trong phạm vi thông tin ngữ cảnh được cung cấp.
"""