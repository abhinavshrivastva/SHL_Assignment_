from together import Together

class QueryRewriter:
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", api_key=None):
        self.client = Together(api_key=api_key)
        self.model = model

    def rewrite(self, query: str) -> str:
        prompt = (
            f"""Your task
is to build the perfect assessment for a candidate applying for the suitable job description. 
Given a job description build the relevant assessment fo the candidate. Do not add any criteria from your own.

The output should follow this format:

{{
  "Assessment Name": "..." 
+ "Description": "...",
}}

Template Responses:
{{
    Assessment Name : C Programming (New) 
    Description : Multi-choice test that measures the knowledge of C programming basics, functions, arrays, composed data types, and advanced C concepts like SLF, file handling and dynamic memory. 
}}
{{
    Assessment Name : Entry Level Cashier Solution
    Description : The Precise Fit Entry Level Cashier Solution is for entry-level retail positions in which employees receive payment in the form of cash, check, or credit cards for goods purchased. Sample tasks for these jobs include, but are not limited to: handling payments, offering customer service, and issuing receipts and refunds.Report Language Availability: English (USA)
}}


Job Description: "{query}"

Output:"""
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
