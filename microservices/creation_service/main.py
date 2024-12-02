import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
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

class CodeRequest(BaseModel):
    requirements: dict

@app.get("/")
async def root():
    return {"message": "Welcome to the Creation Service"}

@app.post("/create_code")
async def create_code(request: CodeRequest):
    prompt = f"Créer une fonction en Python qui {request.requirements['description']}"

    # Charger et optimiser les modèles à la demande
    model_codegen, tokenizer_codegen = await optimize_model("distilgpt2", token)
    model_t5, tokenizer_t5 = await optimize_model("t5-base", token)

    # Appliquer des techniques supplémentaires d'optimisation
    model_codegen = compress_model(model_codegen)
    model_t5 = compress_model(model_t5)
    model_codegen = clusterize_weights(model_codegen)
    model_t5 = clusterize_weights(model_t5)
    model_codegen = use_gpu(model_codegen)
    model_t5 = use_gpu(model_t5)

    def generate_code(prompt, model, tokenizer):
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        outputs = batch_process(model, inputs, batch_size=32)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def review_code(code, model, tokenizer):
        review_prompt = f"Revise the following code for optimization and best practices:\n{code}"
        inputs = tokenizer.encode(review_prompt, return_tensors='pt')
        outputs = batch_process(model, inputs, batch_size=32)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Génération initiale avec DistilGPT-2
    code_1 = generate_code(prompt, model_codegen, tokenizer_codegen)

    # Génération initiale avec T5
    code_2 = generate_code(prompt, model_t5, tokenizer_t5)

    # Revue croisée
    reviewed_code_1 = review_code(code_2, model_codegen, tokenizer_codegen)
    reviewed_code_2 = review_code(code_1, model_t5, tokenizer_t5)

    # Combinaison des résultats
    final_code = f"{reviewed_code_1}\n{reviewed_code_2}"
    return {"code": final_code}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8020))
    uvicorn.run(app, host="0.0.0.0", port=port)
