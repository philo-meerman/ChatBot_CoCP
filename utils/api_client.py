# utils/api_client.py
import requests


def query_huggingface_api(model_id, inputs):
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": "Bearer YOUR_API_KEY"}
    response = requests.post(API_URL, headers=headers, json=inputs)
    return response.json()
