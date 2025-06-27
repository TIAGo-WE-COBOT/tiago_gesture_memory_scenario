""" This script converts the original Keras model for letter classification to a PyTorch model by transferring the weights. The PyTorch architecture was manually defined in `cnn.py` based on `keras_model.summary()` output.

The conversion is a one-time operation, the script is kept for reference only.
"""
import os
import numpy as np
import torch
import tensorflow as tf

from cnn import CNNClassifier

if __name__ == "__main__":    
    # Load model and weights from Keras
    wdir = os.path.abspath(os.path.dirname(__file__))
    keras_model_path = os.path.join(wdir, '..', 'models', 'keras_model.h5')
    keras_model = tf.keras.models.load_model(keras_model_path)

    # Transfer weights from Keras to PyTorch
    pytorch_model_path = os.path.join(wdir, '..', 'models', 'pytorch_weights.pth')
    pytorch_model = CNNClassifier()
    keras_layers = [layer for layer in keras_model.layers if len(layer.get_weights()) > 0]
    torch_layers = [
        pytorch_model.features[0],      # conv2d_2
        pytorch_model.features[3],      # conv2d_3
        pytorch_model.classifier[1],    # dense_4
        pytorch_model.classifier[4],    # dense_5
        pytorch_model.classifier[7],    # dense_6
        pytorch_model.classifier[10],   # dense_7
    ]
    # Sanity check
    assert len(keras_layers) == len(torch_layers), "Mismatch in layer count!"
    for keras_layer, torch_layer in zip(keras_layers, torch_layers):
        weights = keras_layer.get_weights()
        if len(weights) != 2:
            print(f"Skipping layer {keras_layer.name}: unexpected weight structure")
            continue
        w, b = weights
        # Convert weights to tensors and reshape as needed
        if isinstance(torch_layer, torch.nn.Conv2d):
            torch_w = torch.tensor(w).permute(3, 2, 0, 1)  # [H, W, in, out] → [out, in, H, W]
            torch_b = torch.tensor(b)
        elif isinstance(torch_layer, torch.nn.Linear):
            torch_w = torch.tensor(w).T  # [in, out] → [out, in]
            torch_b = torch.tensor(b)
        else:
            print(f"Unsupported layer type: {type(torch_layer)}")
            continue
        # Copy weights safely
        with torch.no_grad():
            torch_layer.weight.copy_(torch_w)
            if torch_layer.bias is not None:
                torch_layer.bias.copy_(torch_b)
    print("All weights transferred!")

    # Save PyTorch weights to disk
    model_path = os.path.join(
        wdir, '..', 'models', "pytorch_weights.pth")
    torch.save(pytorch_model.state_dict(), model_path)
    print(f"Weights saved to {model_path}")