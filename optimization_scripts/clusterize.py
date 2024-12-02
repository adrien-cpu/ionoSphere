def clusterize_weights(model):
    # Implémentation de la clusterisation des poids pour réduire la variance
    # Par exemple, clusteriser les poids en les arrondissant à des valeurs proches.
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = torch.round(param.data * 10) / 10
    return model
