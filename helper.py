import requests
from bs4 import BeautifulSoup 
import re
from src.rewriter import QueryRewriter
from src.retriever import SHLRetriever
from src.reranker import Reranker
from helper import * 

def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n")

        clean_text = re.sub(r'\n\s*\n+', '\n\n', text)
        return clean_text.strip()
    except Exception as e:
        print(f"Error extracting from URL: {e}")
        return "Could not extract text from the provided URL."

def is_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")

def get_recommendation_json(query: str, together_api_key: str):
    print("🔍 Rewriting query...")
    rewriter = QueryRewriter(api_key=together_api_key)
    rewritten_query = rewriter.rewrite(query)
    print(f"\n🔁 Rewritten Query:\n{rewritten_query}")

    print("\n📦 Retrieving candidates...")
    retriever = SHLRetriever()
    results = retriever.retrieve(rewritten_query, k=10)

    print("\n📊 Reranking results...")
    reranker = Reranker()
    ranked = reranker.rerank(rewritten_query, results)

    print("\n✅ Top Results:")
    top_k = ranked.head(10) 
    return {
        "recommended_assessments": [
            {
                "url": row["url"],
                "adaptive_support": "Yes" if row["adaptive"].strip().lower() == "yes" else "No",
                "description": row["description"],
                "duration": int(row["duration"]),
                "remote_support": "Yes" if row["remote_testing"].strip().lower() == "yes" else "No",
                "test_type": row["test_types"] if isinstance(row["test_types"], list) else [row["test_types"]],
            }
            for _, row in top_k.iterrows()
        ]
    }

def recommend_assessments(query: str, together_api_key: str):
    print("🔍 Rewriting query...")
    rewriter = QueryRewriter(api_key=together_api_key)
    rewritten_query = rewriter.rewrite(query)
    print(f"\n🔁 Rewritten Query:\n{rewritten_query}")

    print("\n📦 Retrieving candidates...")
    retriever = SHLRetriever()
    results = retriever.retrieve(rewritten_query, k=10)

    print("\n📊 Reranking results...")
    reranker = Reranker()
    ranked = reranker.rerank(rewritten_query, results)

    print("\n✅ Top Results:")
    return ranked[[
        "name", "url", "remote_testing", "adaptive", 
        "duration", "test_types", "description", "score"
    ]]