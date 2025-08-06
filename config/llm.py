from dotenv import load_dotenv
import os
# llm api
from llama_index.llms.groq import Groq


# Load llm
def load_llm():
    load_dotenv()  # load biến môi trường từ file .env
    API_KEY = os.getenv("MY_API_KEY")
    llm = Groq(model="qwen/qwen3-32b", api_key=API_KEY)
    return llm