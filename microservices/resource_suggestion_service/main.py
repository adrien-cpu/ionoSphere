import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

class ResourceRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Resource Suggestion Service"}

@app.post("/suggest_resources")
async def suggest_resources(request: ResourceRequest):
    # Charger et optimiser les modèles de suggestion de ressources à la demande
    model_bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer_bert.encode(request.query, return_tensors='pt')
    outputs = model_bert(inputs)
    scores = outputs[0][0].tolist()
    return {"resources": scores}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8050))
    uvicorn.run(app, host="0.0.0.0", port=port)
