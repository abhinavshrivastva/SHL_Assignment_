# SHL_Assignment_
🔧 Implementation Details – SHL Assessment Recommendation System
The SHL Assessment Recommendation System is designed as a modular Retrieval-Augmented Generation (RAG) pipeline to generate highly relevant assessment recommendations based on free-form job descriptions or URLs. Here's a detailed breakdown of each component:
🧠 1. Query Rewriting (rewriter.py)
Purpose: Transform the user’s raw input (natural language query or scraped job description) into a clean, focused prompt to guide downstream retrieval and ranking.
Approach:
Utilizes the LLaMA-3.3 70B Instruct Turbo model via Together.ai API.


The model is prompted to gain insight into the job description and rewrite it into a clear, role-specific assessment query.


This rewritten query is not just a reformulation—it’s optimized to describe the ideal assessment for a candidate applying to that job.


Example:
 Input: “Hiring for a frontend developer with JavaScript and HTML skills. Duration under 30 minutes.”
 Rewritten: “Find a cognitive or technical assessment for a frontend web developer with proficiency in JavaScript, HTML, and UI development, preferably short duration.”
This step lays the foundation for accurate retrieval.

📦 2. Retriever (retriever.py)
Purpose: Surface a diverse and relevant set of candidate assessments from the SHL catalog using vector-based similarity.
Approach:
SHL assessments are indexed using sentence-transformer embeddings.


The rewritten query is encoded and compared via cosine similarity to rank the top 10 candidates.


This step ensures coverage, bringing in assessments across domains that could be relevant.


Key Insight:
The retriever acts as a comparison layer—by surfacing assessments with similar contexts, it allows downstream ranking models to perform intelligent matching.



📊 3. Reranker (reranker.py)
Purpose: Score and reorder the top-k retrieved assessments based on how well they match the rewritten query.
Approach:
Uses the cross-encoder/ms-marco-MiniLM-L-6-v2 transformer-based model.


Each query-assessment pair is passed through the cross-encoder for deep semantic alignment.


The model outputs a relevance score, which is used to sort and return the top results.




