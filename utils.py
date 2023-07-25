
def sequential(layers, x):
    for layer in layers:
        x = layer(x)
    return x
