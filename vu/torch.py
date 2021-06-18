import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import z3

DBG = pdb.set_trace


class Model_Torch4(nn.Module):
    def forward(self, x):
        W1 = torch.Tensor([[1.0, 1.0], [-1.0, 1.0]])
        b1 = torch.Tensor([0.0, 0.0])
        x = x @ W1 + b1
        x = F.relu(x)

        W2 = torch.Tensor([[0.5, -0.5], [-0.2, 0.1]])
        b2 = torch.Tensor([0.0, 0.0])
        x = x @ W2 + b2
        x = F.relu(x)

        W3 = torch.Tensor([[1.0, -1.0], [-1.0, 1.0]])
        b3 = torch.Tensor([0.0, 0.0])
        x = x @ W3 + b3

        return x


class Model_Torch(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):

        for (weights, biases, is_relu) in self.layers:
            x = x @ weights + biases
            if is_relu:
                x = F.relu(x)
        return x


print("TORCH")
model_pa4_torch_a = Model_Torch(None)
print(model_pa4_torch_a)


def model_pa4_torch():
    W1 = torch.Tensor([[1.0, 1.0], [-1.0, 1.0]])
    b1 = torch.Tensor([0.0, 0.0])
    is_relu1 = True
    layer1 = (W1, b1, is_relu1)

    W2 = torch.Tensor([[0.5, -0.5], [-0.2, 0.1]])
    b2 = torch.Tensor([0.0, 0.0])
    is_relu2 = True
    layer2 = (W2, b2, is_relu2)

    W3 = torch.Tensor([[1.0, -1.0], [-1.0, 1.0]])
    b3 = torch.Tensor([0.0, 0.0])
    is_relu3 = False
    layer3 = (W3, b3, is_relu3)
    layers = [layer1, layer2, layer3]

    return Model_Torch(layers)


model4 = model_pa4_torch()
print(model4)
