Here’s a clean and professional `README.md` for your SHL Assessment Recommendation System:

---

# 🚀 SHL Assessment Recommendation System

A modular Retrieval-Augmented Generation (RAG) pipeline that recommends relevant SHL assessments based on free-form job descriptions or URLs.

---

## 🔧 Overview

This system intelligently transforms raw job inputs into role-specific queries, retrieves relevant SHL assessments, and reranks them using deep semantic matching—all optimized for accuracy, diversity, and relevance.

---

## 📁 Components

### 🧠 1. Query Rewriting [`rewriter.py`]

**Purpose:**  
Transform raw user input (free-form job description or scraped URL content) into a structured, role-specific query optimized for assessment retrieval.

**Approach:**
- Powered by the **LLaMA-3.3 70B Instruct Turbo** model via the Together.ai API.
- Reformulates the input into a clear, targeted prompt that reflects the ideal assessment characteristics.

**Example:**  
> **Input:** "Hiring for a frontend developer with JavaScript and HTML skills. Duration under 30 minutes."  
> **Rewritten:** "Find a cognitive or technical assessment for a frontend web developer with proficiency in JavaScript, HTML, and UI development, preferably short duration."

---

### 📦 2. Retriever [`retriever.py`]

**Purpose:**  
Retrieve a diverse set of SHL assessments that are semantically relevant to the rewritten query.

**Approach:**
- All SHL assessments are embedded using **sentence-transformer** models.
- Rewritten query is encoded and compared using **cosine similarity**.
- Returns the top 10 most similar assessments.

**Key Insight:**  
Acts as a **broad coverage layer**—ensures the system doesn’t miss potentially relevant assessments by casting a wide net.

---

### 📊 3. Reranker [`reranker.py`]

**Purpose:**  
Rank the top-k retrieved assessments based on their true semantic relevance to the query.

**Approach:**
- Uses **`cross-encoder/ms-marco-MiniLM-L-6-v2`**, a transformer-based cross-encoder.
- Evaluates query-assessment pairs for deep contextual alignment.
- Outputs relevance scores used to return the most precise recommendations.

---

## ✅ Output

A list of **up to 10 ranked SHL assessments** tailored to the job description, optimized for fit and utility.

---

## 📌 Tech Stack

- **LLMs:** LLaMA-3.3 70B via Together.ai
- **Embeddings:** Sentence Transformers
- **Semantic Ranking:** Cross-Encoder (MS MARCO)
- **Language:** Python 3.10+
- **Framework:** Modular, pluggable RAG components

---

## 📬 Contact

For questions or contributions, feel free to reach out or raise an issue.

---
