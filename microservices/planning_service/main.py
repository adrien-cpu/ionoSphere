import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

class PlanRequest(BaseModel):
    prompt: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Planning Service"}

@app.post("/plan")
async def create_plan(request: PlanRequest):
    # Charger et optimiser les modèles de planification à la demande
    model_distilbert = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer_distilbert = AutoTokenizer.from_pretrained("distilgpt2")

    inputs = tokenizer_distilbert.encode(request.prompt, return_tensors='pt')
    outputs = model_distilbert.generate(inputs, max_length=100)

    plan = tokenizer_distilbert.decode(outputs[0], skip_special_tokens=True)
    return {"plan": plan}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8040))
    uvicorn.run(app, host="0.0.0.0", port=port)
