def batch_process(model, inputs, batch_size=32):
    outputs = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        with torch.no_grad():
            batch_output = model(batch)
        outputs.extend(batch_output)
    return outputs
