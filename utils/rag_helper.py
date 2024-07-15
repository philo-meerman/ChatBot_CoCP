# utils/rag_helper.py
from models.rag_model import RAGModel

rag_model = RAGModel()


def generate_answer(query, context):
    return rag_model.generate_response(query, context)
