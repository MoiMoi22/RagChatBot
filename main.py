from fastapi import FastAPI
from api.end_points import router as ask_router

app = FastAPI(
    title="RAG Department QA",
    description="API truy vấn tài liệu nội bộ theo phòng ban, có router thông minh",
    version="1.0.0"
)

# Register endpoints
app.include_router(ask_router, prefix="")

# Optional: Health check
@app.get("/ping")
async def ping():
    return {"message": "pong"}
