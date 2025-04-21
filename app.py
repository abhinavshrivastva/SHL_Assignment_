import os
from helper import * 
from dotenv import load_dotenv
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


if __name__ == "__main__":
     
    user_query = "Hiring for a frontend developer. Duration under 30 minutes."
    if is_url(user_query):
        print("🌐 Detected URL. Extracting job description...")
        query = extract_text_from_url(user_query)
        print(f"\n📝 Extracted Job Description:\n{query}...")  # show a preview

    top_results = recommend_assessments(user_query, TOGETHER_API_KEY)
    print(top_results.to_markdown(index=True))


