import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.prune import l1_unstructured

def optimize_model(model_name, token):
    # Charger le modèle
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    
    # Appliquer la quantization dynamique
    model_quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Pruning des couches linéaires du modèle
    for name, module in model_quantized.named_modules():
        if isinstance(module, torch.nn.Linear):
            l1_unstructured(module, name='weight', amount=0.4)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    return model_quantized, tokenizer
