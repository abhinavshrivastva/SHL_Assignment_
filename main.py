# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.rewriter import QueryRewriter
from src.retriever import SHLRetriever
from src.reranker import Reranker
from helper import get_recommendation_json
from dotenv import load_dotenv
import os

# Load API key from environment
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# FastAPI instance
app = FastAPI(title="SHL Assessment Recommendation API")


# Request body model
class QueryInput(BaseModel):
    query: str


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/recommend")
def recommend(input: QueryInput):
    if not TOGETHER_API_KEY:
        return {"error": "Missing TOGETHER_API_KEY"}
    return get_recommendation_json(input.query, TOGETHER_API_KEY)
