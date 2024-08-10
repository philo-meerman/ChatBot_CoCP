import sys
import os
import re
import json
import logging
from datetime import datetime  # Add this import

# Suppress logging messages below CRITICAL level
logging.basicConfig(level=logging.CRITICAL)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

# Add the root of the project to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.answer_generator import generate_answer
from init.init_rag_model import initialize_rag_model
from config import Config


def load_json(filepath):
    """
    Load a JSON file from the specified filepath.

    Args:
    filepath (str): The path to the JSON file.

    Returns:
    dict: The content of the JSON file as a dictionary.
    """
    with open(filepath, "r") as file:
        return json.load(file)


def calculate_tfidf_similarity(generated_response, expected_response):
    """
    Calculate the cosine similarity between two text responses using TF-IDF vectors.

    Args:
    generated_response (str): The generated response text.
    expected_response (str): The expected response text.

    Returns:
    float: The cosine similarity between the two responses.
    """
    vectorizer = TfidfVectorizer().fit_transform(
        [generated_response, expected_response]
    )
    vectors = vectorizer.toarray()
    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return round(similarity, 2)


def calculate_llm_similarity(generated_response, expected_response, llm):
    """
    Calculate the similarity between two text responses using an LLM.

    Args:
    generated_response (str): The generated response text.
    expected_response (str): The expected response text.
    llm (OpenAI): The LLM instance to use for similarity calculation.

    Returns:
    float: The similarity score between the two responses.
    """
    # Define the prompt for the LLM
    prompt = f"""
    Compare the following two responses and provide a similarity score between 0 and 1, where 1 means they are identical in meaning and 0 means they are completely different in meaning.

    Response 1: {generated_response}

    Response 2: {expected_response}

    Return only the similarity score as a float:
    """

    # Prepare the messages for the LLM
    messages = [
        ChatMessage(
            role="system",
            content="You are an expert assistant helping to compare text similarities.",
        ),
        ChatMessage(role="system", content=prompt),
    ]

    response = llm.chat(messages)

    # Extract the similarity score from the response
    text = response.message.content.strip()
    pattern = r"\d+\.\d+"
    match = re.search(pattern, text)
    if match:
        similarity_score = float(match.group())
    else:
        similarity_score = 0.0  # Default to 0 if parsing fails

    return similarity_score


def main():
    """
    Main function to run test queries, compare responses, and output results.

    This function generates responses for the test queries, compares them with the ground truth,
    and outputs the results both to the console and to a JSON file for further analysis.
    """
    # Initialize the RAG model
    rag_model = initialize_rag_model()

    # Initialize the LLM
    llm = OpenAI(temperature=0, model=Config.CHAT_MODEL, max_tokens=Config.MAX_TOKENS)

    # Load the ground truth queries and responses
    ground_truth = load_json("evaluation/ground_truth.json")

    # Run the queries and store the responses
    results = []

    for item in ground_truth:
        query = item["query"]
        expected_response = item["response"]
        generated_response, _ = generate_answer(query, rag_model)

        # Calculate similarity using TFIDF
        tfidf_similarity_score = calculate_tfidf_similarity(
            generated_response, expected_response
        )

        # Calculate similarity using LLM
        llm_similarity_score = calculate_llm_similarity(
            generated_response, expected_response, llm
        )

        # Check if either similarity score is sufficient
        is_sufficient = tfidf_similarity_score >= 0.7 or llm_similarity_score >= 0.7

        results.append(
            {
                "query": query,
                "ground_truth": expected_response,
                "generated_response": generated_response,
                "tfidf_similarity_score": tfidf_similarity_score,
                "llm_similarity_score": llm_similarity_score,
                "is_sufficient": bool(
                    is_sufficient
                ),  # Ensure the sufficiency flag is a bool
            }
        )

    # Print summary of results
    sufficient_count = sum(result["is_sufficient"] for result in results)
    total_count = len(results)
    print(
        f"Results: {sufficient_count}/{total_count} responses are above the similarity threshold of 0.7"
    )

    # Print the detailed results in a nicely readable format
    print(json.dumps(results, indent=4, ensure_ascii=False))

    # Generate a timestamp and format it for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output the results to a JSON file with the timestamp in the filename
    output_filename = f"evaluation/comparison_results_{timestamp}.json"
    with open(output_filename, "w") as outfile:
        json.dump(results, outfile, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
