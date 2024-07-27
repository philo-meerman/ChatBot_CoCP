import sys
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.pdf_processor import extract_boek_2_text
from utils.rag_helper import generate_answer
from models.rag_model import RAGModel
load_dotenv()

app = Flask(__name__)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

rag_model = None

def initialize_rag_model():
    global rag_model

    pdf_path="data/vm1kkye15yy2.pdf"
    text = extract_boek_2_text(pdf_path=pdf_path)
    regenerate_embeddings = 0

    rag_model = RAGModel(api_key=api_key)
    rag_model.chunk_text(text)
    if regenerate_embeddings == 1:
        rag_model.generate_embeddings()
        rag_model.store_embeddings()
    rag_model.load_embeddings()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = generate_answer(query=user_input, rag_model=rag_model, api_key=api_key)
    return jsonify({"response": response})

if __name__ == "__main__":
    initialize_rag_model()
    app.run(debug=True)
