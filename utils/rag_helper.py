# rag_helper.py

import openai
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI


def generate_answer(query, rag_model, api_key, chat_model="gpt-3.5-turbo", max_tokens=1024):
    openai.api_key = api_key

    relevant_chunks = rag_model.get_relevant_chunks(query, k=5)
    context = "\n\n".join(relevant_chunks)

    prompt = f"Answer this query: {query}\n\nContext:\n{context}\n\nAnswer:"

    messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content=prompt),
        ]
    
    llm = OpenAI(temperature=0, model=chat_model, max_tokens=max_tokens)

    responses = llm.chat(messages)

    answer = responses.message.content
    return answer
