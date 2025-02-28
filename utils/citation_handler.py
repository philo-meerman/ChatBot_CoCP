"""
Citation Handler Module

This module provides utility functions to process citation-related queries within the
ChatBot_DPC project. It is designed to detect direct citation requests, extract article
numbers from text, and retrieve the exact citation or relevant citation chunks using a
Retrieval Augmented Generation (RAG) model.

Functions:
    is_direct_citation_request(query):
        Checks if the user's query contains keywords that indicate a direct request for a citation.

    extract_article_numbers(text):
        Extracts article numbers from the given text using regular expressions.

    get_direct_citation(query, rag_model, conversation_history, k=1):
        Attempts to retrieve the exact citation from the RAG model by first extracting article
        numbers from the query (or, if necessary, from the conversation history). If an exact
        match is found, it returns the corresponding article content; otherwise, it falls back
        to performing a general search for relevant citation chunks.

Dependencies:
    - re: For performing regular expression operations.
    - logging: For logging the process and results of citation extraction and retrieval.

Usage:
    >>> from citation_handler import is_direct_citation_request, extract_article_numbers,
            get_direct_citation
    >>> if is_direct_citation_request(query):
    ...     citation = get_direct_citation(query, rag_model, conversation_history)
"""

import re
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Keywords and patterns to detect direct citation requests
CITATION_KEYWORDS = ["citeer", "citaat", "exacte tekst", "exacte artikel"]


def is_direct_citation_request(query):
    """
    Check if the query is a direct request for a citation.

    Parameters:
    - query (str): The user's query.

    Returns:
    - bool: True if the query requests a direct citation, False otherwise.
    """
    logging.info("Checking if the query is a direct citation request.")
    for keyword in CITATION_KEYWORDS:
        if re.search(keyword, query, re.IGNORECASE):
            logging.info(
                "Keyword '%s' found in the query. It's a direct citation request.",
                keyword,
            )
            return True
    logging.info("No direct citation keywords found in the query.")
    return False


def extract_article_numbers(text):
    """
    Extract all article numbers from the text.

    Parameters:
    - text (str): The input text.

    Returns:
    - list: A list of extracted article numbers.
    """
    logging.info("Extracting article numbers from the text.")
    article_numbers = re.findall(r"Artikel\s+(\d+\.\d+\.\d+)", text, re.IGNORECASE)
    logging.info("Extracted article numbers: %s", article_numbers)
    return article_numbers


def get_direct_citation(query, rag_model, conversation_history, k=1):
    """
    Retrieve the exact citation from the RAG model.

    Parameters:
    - query (str): The user's query.
    - rag_model: The RAG model instance.
    - conversation_history (list): The conversation history.
    - k (int): Number of top similar chunks to retrieve.

    Returns:
    - citation (str): The exact citation text if found, otherwise an error message.
    """
    logging.info("Attempting to retrieve the exact citation based on the query.")

    article_numbers = extract_article_numbers(query)

    if not article_numbers:
        logging.info(
            "No article numbers found in the current query. Checking conversation history."
        )

        # Log the conversation history only if it is being checked
        logging.info("Current conversation history:")
        for msg in conversation_history:
            logging.info("%s: %s", msg.role, msg.content)

        # Check the last response for article numbers
        for msg in reversed(conversation_history):
            if msg.role == "assistant":
                article_numbers = extract_article_numbers(msg.content)
                if article_numbers:
                    logging.info(
                        "Article numbers found in conversation history: %s",
                        article_numbers,
                    )
                    break

    if article_numbers:
        logging.info(
            "Found article numbers: %s. Retrieving exact citation.", article_numbers
        )
        for num in article_numbers:
            article_content = rag_model.get_exact_article(num)
            if article_content:
                logging.info("Retrieved content for article %s.", num)
                return (
                    article_content
                    if isinstance(article_content, str)
                    else article_content
                )
            logging.warning("No content found for article %s.", num)

    logging.info("No specific article number found. Falling back to general search.")
    relevant_chunks, _ = rag_model.get_relevant_chunks(query, k)
    if relevant_chunks:
        context = "\n\n".join(relevant_chunks)
        logging.info("Relevant chunks found and compiled into a citation.")
        return f"Hier is het gevraagde citaat:\n\n{context}"
    logging.error("No relevant chunks found for the query.")
    return "Sorry, ik kon het gevraagde citaat niet vinden in het document."
