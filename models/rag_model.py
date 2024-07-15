# models/rag_model.py
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration


class RAGModel:
    def __init__(self):
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
        self.model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

    def generate_response(self, query, context):
        inputs = self.tokenizer(query, return_tensors="pt")
        generated = self.model.generate(
            context_input_ids=inputs["input_ids"],
            context_attention_mask=inputs["attention_mask"],
            num_beams=2,
            max_length=64,
            min_length=10,
            early_stopping=True,
        )
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return response
