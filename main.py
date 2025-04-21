from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from src.rewriter import QueryRewriter
from src.retriever import SHLRetriever
from src.reranker import Reranker
from helper import * 
from dotenv import load_dotenv
import os
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

app = FastAPI()


class QueryInput(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(input: QueryInput):
    return get_recommendation_json(input.query, TOGETHER_API_KEY)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

