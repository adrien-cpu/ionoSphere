import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

class StrategyRequest(BaseModel):
    description: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Strategy Service"}

@app.post("/plan_strategy")
async def plan_strategy(request: StrategyRequest):
    # Charger et optimiser les modèles de stratégie à la demande
    model_xlnet = AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased")
    tokenizer_xlnet = AutoTokenizer.from_pretrained("xlnet-base-cased")
    model_electra = AutoModelForSequenceClassification.from_pretrained("google/electra-base-discriminator")
    tokenizer_electra = AutoTokenizer.from_pretrained("google/electra-base-discriminator")

    def analyze_with_model(text, model, tokenizer):
        inputs = tokenizer.encode(text, return_tensors='pt')
        outputs = model(inputs)
        scores = outputs[0][0].tolist()
        return scores

    # Analyse croisée avec XLNet et Electra
    scores_xlnet = analyze_with_model(request.description, model_xlnet, tokenizer_xlnet)
    scores_electra = analyze_with_model(request.description, model_electra, tokenizer_electra)

    # Agrégation des résultats
    final_strategy = {"xlnet_scores": scores_xlnet, "electra_scores": scores_electra}

    return {"strategy": final_strategy}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 800))
    uvicorn.run(app, host="0.0.0.0", port=port)
