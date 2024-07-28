import re

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
    for keyword in CITATION_KEYWORDS:
        if re.search(keyword, query, re.IGNORECASE):
            return True
    return False

def extract_article_numbers(text):
    """
    Extract all article numbers from the text.

    Parameters:
    - text (str): The input text.

    Returns:
    - list: A list of extracted article numbers.
    """
    return re.findall(r'Artikel\s+(\d+\.\d+\.\d+)', text, re.IGNORECASE)

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
    article_numbers = extract_article_numbers(query)
    
    # Check if article numbers are in the current query
    if not article_numbers:
        # Check the last response for article numbers
        for msg in reversed(conversation_history):
            if msg.role == 'assistant':
                article_numbers = extract_article_numbers(msg.content)
                if article_numbers:
                    break
    
    if article_numbers:
        for num in article_numbers:
            article_content = rag_model.get_exact_article(num)
            if article_content:
                return article_content if isinstance(article_content, str) else article_content
    
    # If no specific article number is found, fall back to general search
    relevant_chunks = rag_model.get_relevant_chunks(query, k)
    if relevant_chunks:
        context = "\n\n".join(relevant_chunks)
        return f"Hier is het gevraagde citaat:\n\n{context}"
    else:
        return "Sorry, ik kon het gevraagde citaat niet vinden in het document."