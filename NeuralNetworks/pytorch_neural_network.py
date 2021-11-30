import torch
import pandas as pd
import numpy as np


class PyTorchBinaryClassifier(torch.nn.Module):
    '''https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89'''

    def __init__(self, input_count: int, node_counts: list, activation=torch.nn.ReLU()) -> None:
        super(PyTorchBinaryClassifier, self).__init__()
        self.activation = activation
        self.layers = self.generate_layers(input_count, node_counts)
        self.layer_1 = torch.nn.Linear(input_count, node_counts[0])
        self.layer_out = torch.nn.Linear(node_counts[0], 1)

    def generate_layers(self, input_count, node_counts) -> list:
        layers = list()
        for node_count in node_counts:
            layers.append(torch.nn.Linear(input_count, node_count))
            input_count = node_count
        layers.append(torch.nn.Linear(node_count, 1))
        return layers

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z = X
        for layer in self.layers[:-1]:
            Z = self.activation(layer(Z))
        return self.layers[-1](Z)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
