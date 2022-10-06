import requests
import json
from embedders.extraction.contextual import TransformerTokenEmbedder

embedder = TransformerTokenEmbedder('distilbert-base-uncased', "en_core_web_sm")

def get_container_response(text_input, aci_domain):
    transformed_input = embedder.transform([text_input])
    
    # Convert to dict, then to JSON
    input_dict = {"inputs": transformed_input}
    with open("data.json", "w") as fp:
        json.dump(input_dict, fp)
    with open("data.json") as f:
        data = f.read().replace("\n", "")

    # Headers to send out requests
    headers = {
        "Content-Type": "application/json"
    }

    # Send post request
    response = requests.post(f"http://{aci_domain}:8080/v1/models/model:predict", headers=headers, data=data)
    return response