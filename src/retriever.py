import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class SHLRetriever:
    def __init__(self, 
                 csv_path="data/combined_shl_assessments.csv", 
                 model_name="all-MiniLM-L6-v2", 
                 index_path="data/index.faiss"):
        self.df = pd.read_csv(csv_path)
        self.df["context"] = (
    "Assessment Name: " + self.df["name"].fillna('') + ". " +
    "Description: " + self.df["description"].fillna('') + "." 

)
        self.df["meta_data"] = self.df.apply(build_metadata, axis=1)
        self.model = SentenceTransformer(model_name)

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.build_index(index_path)

    def build_index(self, index_path):
        embeddings = self.model.encode(self.df["context"].tolist(), show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))
        faiss.write_index(self.index, index_path)

    def retrieve(self, query: str, k: int = 10):
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(np.array(query_embedding), k)
        return self.df.iloc[indices[0]].copy()


def build_metadata(row):
    parts = []
    if pd.notna(row["test_types"]): parts.append(f"Type: {row['test_types']}")
    if pd.notna(row["duration"]): parts.append(f"Duration: {row['duration']}")
    if pd.notna(row["remote_testing"]): parts.append(f"Remote Testing: {row['remote_testing']}")
    if pd.notna(row["adaptive"]): parts.append(f"Adaptive/IRT: {row['adaptive']}")
    return ". ".join(parts) + "."
