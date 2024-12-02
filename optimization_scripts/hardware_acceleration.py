import torch

def use_gpu(model):
    if torch.cuda.is_available():
        model = model.cuda()
    return model
