from sentence_transformers import CrossEncoder
import pandas as pd

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L12-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        pairs = [(query, context) for context in results_df["context"]]
        scores = self.model.predict(pairs)
        results_df["score"] = scores
        return results_df.sort_values("score", ascending=False).head(top_k)

