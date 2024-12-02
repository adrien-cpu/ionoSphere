import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from optimization_scripts.quantize_prune import optimize_model
from optimization_scripts.compress import compress_model
from optimization_scripts.clusterize import clusterize_weights
from optimization_scripts.hardware_acceleration import use_gpu
from optimization_scripts.batching import batch_process

app = FastAPI()

# Authentification avec Hugging Face
token = os.getenv("HUGGINGFACE_TOKEN")
if token is None:
    raise EnvironmentError("HUGGINGFACE_TOKEN not set")

# Ajout de la route pour le favicon
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

class AnalysisRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Analysis Service"}

@app.post("/analyze")
async def analyze_text(request: AnalysisRequest):
    # Charger et optimiser les modèles d'analyse à la demande
    model_bert, tokenizer_bert = await optimize_model("bert-base-uncased", token)
    model_roberta, tokenizer_roberta = await optimize_model("roberta-base", token)
    
    # Appliquer des techniques supplémentaires d'optimisation
    model_bert = compress_model(model_bert)
    model_roberta = compress_model(model_roberta)
    model_bert = clusterize_weights(model_bert)
    model_roberta = clusterize_weights(model_roberta)
    model_bert = use_gpu(model_bert)
    model_roberta = use_gpu(model_roberta)

    def analyze_with_model(text, model, tokenizer):
        inputs = tokenizer.encode(text, return_tensors='pt')
        outputs = batch_process(model, inputs, batch_size=32)
        scores = outputs[0][0].tolist()
        return scores

    # Analyse croisée avec BERT et RoBERTa
    scores_bert = analyze_with_model(request.text, model_bert, tokenizer_bert)
    scores_roberta = analyze_with_model(request.text, model_roberta, tokenizer_roberta)

    # Agrégation des résultats
    final_scores = {"bert_scores": scores_bert, "roberta_scores": scores_roberta}
    
    return {"scores": final_scores}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
