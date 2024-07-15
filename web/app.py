# web/app.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request, jsonify
from utils.pdf_processor import extract_text_from_pdf
from utils.rag_helper import generate_answer

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    context = extract_text_from_pdf(
        "data/tk-memorie-van-toelichting-nieuw-wetboek-van-sv.pdf"
    )
    response = generate_answer(user_input, context)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
