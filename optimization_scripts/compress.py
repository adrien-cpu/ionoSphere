import torch

def compress_model(model):
    # Appliquer des techniques de compression telles que la réduction de précision, la factorisation de matrice, etc.
    # Par exemple, vous pouvez appliquer la quantization statique ici :
    model_compressed = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model_compressed
