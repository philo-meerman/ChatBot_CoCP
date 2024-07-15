# utils/rag_helper.py
import warnings
from transformers import pipeline

# Initialize the pipeline once at the module level
pipe = pipeline("fill-mask", model="pdelobelle/robbert-v2-dutch-base")


def generate_answer(query, context):
    # Ensure the query includes the mask token
    if "<mask>" not in query:
        query += "<mask>"

    # Debug: Print the query being sent to the pipeline
    print(f"Query to pipeline: {query}")  

    # Use the pipeline to generate a response
    result = pipe(query)

    # Debug: Print the result from the pipeline
    print(f"Pipeline result: {result}")

    # Custom logic to select the most appropriate answer
    for item in result:
        token_str = item["token_str"].strip()
        # Filter out punctuation and select appropriate token
        if token_str.isalpha():
            return item["sequence"]
    

    
    return result[0]["sequence"]  # Adjust based on the pipeline output
