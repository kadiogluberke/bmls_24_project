from fastapi import FastAPI
import uvicorn
from backend.api import router as api_v1_router

app = FastAPI()

app.include_router(api_v1_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(app)
