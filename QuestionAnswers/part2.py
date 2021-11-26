from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import NeuralNetworks as nn


def q2a():
    print('\nQ2a')
    X = pd.DataFrame(data={
        'x1': [1],
        'x2': [1],
    })
    y = pd.Series([1])
    network = nn.NeuralNetwork(X.columns.tolist())

    layer1 = nn.Layer(3, 2)
    layer1.W = np.array(
        [
            [-1, -2, -3],
            [1, 2, 3]
        ]
    )
    # print(layer1.W)
    # x = np.array([1, 1, 1])
    z1 = layer1.eval(np.array([1, 1, 1]))
    # print(z1)

    layer2 = nn.Layer(3, 2)
    layer2.W = np.array(
        [
            [-1, -2, -3],
            [1, 2, 3]
        ]
    )
    # print(layer2.W)
    z2 = layer2.eval(z1)
    # print(z2)

    layer3 = nn.OutputLayer(3, 1)
    layer3.W = np.array([-1, 2, -1.5])
    # print(layer3.W)
    y = layer3.eval(z2)
    # print(y)

    network.add_hidden_layers([layer1, layer2])
    network.add_output_layer(layer3)

    # print(network.eval(X.copy()))

    network.calc_gradient(X.copy(), 1)

    # Results above indicate that evaluation works; now for back propagation . . .


def q2b():
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')


def load_bank_note_data(csv: str) -> (pd.DataFrame, pd.Series):
    X_cols = ['WaveletVariance', 'WaveletSkew', 'WaveletCurtosis', 'ImageEntropy']
    y_col = 'Label'

    train = load_data(csv)
    X = train[X_cols]
    y = encode_vals(train[y_col])
    return X, y


def load_data(csv: str) -> pd.DataFrame:
    return pd.read_csv(
        Path('bank-note', 'bank-note', csv),
        names=['WaveletVariance', 'WaveletSkew', 'WaveletCurtosis', 'ImageEntropy', 'Label']
    )


def encode_vals(y: pd.Series) -> pd.Series:
    return y.replace({0: -1})
