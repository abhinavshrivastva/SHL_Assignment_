import streamlit as st
from src.rewriter import QueryRewriter
from src.retriever import SHLRetriever
from src.reranker import Reranker
import pandas as pd
import os
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommendation System")

# User input
query = st.text_area("Enter Job Description or Query", 
                     placeholder="e.g. Hiring a frontend developer with React and JavaScript experience.")

if st.button("🔎 Recommend Assessments") and query.strip():
    with st.spinner("Running RAG pipeline..."):
        try:
            
            rewriter = QueryRewriter(api_key=TOGETHER_API_KEY)
            rewritten_query = rewriter.rewrite(query)

            retriever = SHLRetriever()
            results = retriever.retrieve(rewritten_query, k=10)

            reranker = Reranker()
            ranked = reranker.rerank(rewritten_query, results)

            top_k = ranked.head(10)
            display_df = top_k[[
                "name", "url", "description", "duration", 
                "remote_testing", "adaptive", "test_types", "score"
            ]]
            display_df = display_df.rename(columns={
                "name": "Assessment Name",
                "url": "Link",
                "description": "Description",
                "duration": "Duration (min)",
                "remote_testing": "Remote Support",
                "adaptive": "Adaptive Support",
                "test_types": "Test Types",
                "score": "Relevance Score"
            })

            st.success("✅ Recommendations Ready")
            st.dataframe(display_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
