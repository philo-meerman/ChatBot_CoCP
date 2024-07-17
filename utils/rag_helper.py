# utils/rag_helper.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_answer(query, context):
    # Encode the user input and context
    inputs = tokenizer.encode_plus(
        query + tokenizer.eos_token, return_tensors="pt", padding=True, truncation=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate a response
    response_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the response
    response = tokenizer.decode(
        response_ids[:, input_ids.shape[-1] :][0], skip_special_tokens=True
    )

    return response
