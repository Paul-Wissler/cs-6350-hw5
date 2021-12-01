from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

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
    print(layer1.W)
    z1 = layer1.eval(np.array([1, 1, 1]))
    print(z1)

    layer2 = nn.Layer(3, 2)
    layer2.W = np.array(
        [
            [-1, -2, -3],
            [1, 2, 3]
        ]
    )
    print(layer2.W)
    z2 = layer2.eval(z1)
    print(z2)

    layer3 = nn.OutputLayer(3, 1)
    layer3.W = np.array([[-1, 2, -1.5]])
    print(layer3.W)
    y = layer3.eval(z2)
    print(y)

    network.add_hidden_layers([layer1, layer2])
    network.add_output_layer(layer3)

    print(network.eval(X.copy()))

    network.calc_gradient(X.copy(), 1)

    # Results above indicate that evaluation works; now for back propagation . . .


def q2b():
    print('\nQ2b')
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')

    node_counts = [
        5,
        10,
        25,
        50,
        100,
    ]
    for node_count in node_counts:
        network = nn.NeuralNetwork(X.columns.tolist(), random_seed=False, rate=0.1, rate_damping=10)

        layer1 = nn.Layer(5, node_count - 1, name='layer1', inst='gauss')
        layer2 = nn.Layer(node_count, node_count - 1, name='layer2', inst='gauss')
        layer3 = nn.OutputLayer(node_count, 1, name='layer3', inst='gauss')
    
        network.add_hidden_layers([layer1, layer2])
        network.add_output_layer(layer3)

        network.train(X.copy(), y.copy(), epochs=10)
        print('\nLAYERS', node_count)
        print('TRAINING ERROR:', network.test(X.copy(), y.copy()))
        print('TEST ERROR:', network.test(X_test.copy(), y_test.copy()))
    
    # Q2b

    # LAYERS 5
    # TRAINING ERROR: 0.9357798165137615
    # TEST ERROR: 0.93

    # LAYERS 10
    # TRAINING ERROR: 0.9736238532110092
    # TEST ERROR: 0.968
    # C:\Users\pwiss\OneDrive\Documents\GitHub\cs-6350-hw5\NeuralNetworks\activation_functions.py:5: RuntimeWarning: overflow encountered in exp
    #   return 1 / (1 + np.exp(-x))

    # LAYERS 25
    # TRAINING ERROR: 0.9518348623853211
    # TEST ERROR: 0.946

    # LAYERS 50
    # TRAINING ERROR: 0.9403669724770642
    # TEST ERROR: 0.952

    # LAYERS 100
    # TRAINING ERROR: 0.6410550458715596
    # TEST ERROR: 0.642


def q2c():
    print('\nQ2c')
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')

    node_counts = [
        5,
        10,
        25,
        50,
        100,
    ]
    for node_count in node_counts:
        network = nn.NeuralNetwork(X.columns.tolist(), random_seed=False, rate=0.1, rate_damping=10)

        layer1 = nn.Layer(5, node_count - 1, name='layer1', inst='zero')
        layer2 = nn.Layer(node_count, node_count - 1, name='layer2', inst='zero')
        layer3 = nn.OutputLayer(node_count, 1, name='layer3', inst='zero')
    
        network.add_hidden_layers([layer1, layer2])
        network.add_output_layer(layer3)

        network.train(X.copy(), y.copy(), epochs=10)
        print('\nLAYERS', node_count)
        print('TRAINING ERROR:', network.test(X.copy(), y.copy()))
        print('TEST ERROR:', network.test(X_test.copy(), y_test.copy()))

    # Q2c

    # LAYERS 5
    # TRAINING ERROR: 0.786697247706422
    # TEST ERROR: 0.794

    # LAYERS 10
    # TRAINING ERROR: 0.9288990825688074
    # TEST ERROR: 0.928

    # LAYERS 25
    # TRAINING ERROR: 0.9357798165137615
    # TEST ERROR: 0.934

    # LAYERS 50
    # TRAINING ERROR: 0.948394495412844
    # TEST ERROR: 0.932

    # LAYERS 100
    # TRAINING ERROR: 0.8692660550458715
    # TEST ERROR: 0.856


def q2e():
    print('\nQ2e')

    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')
    X['MODEL_BIAS'] = 1
    X_test['MODEL_BIAS'] = 1
    X = torch.tensor(X.values, dtype=torch.double)
    X_test = torch.tensor(X_test.values, dtype=torch.double)
    y = torch.tensor(y.values, dtype=torch.double)
    y_test = torch.tensor(y_test.values, dtype=torch.double)

    depths = [
        3,
        5,
        9,
    ]
    widths = [
        5,
        10,
        25,
        50,
        100,
    ]
    epoch_count = 1
    
    he = torch.nn.init.kaiming_uniform_
    xavier = torch.nn.init.xavier_uniform_
    init_types = [he, xavier]
    activations = [torch.nn.Tanh(), torch.nn.ReLU()]
    models = dict()
    output_df = pd.DataFrame()
    for init_type, activation in zip(init_types, activations):
        print('')
        print(init_type, activation)
        for depth in depths:
            for width in widths:
                model3 = torch.nn.Sequential(
                        torch.nn.Linear(5, width),
                        activation,
                        torch.nn.Linear(width, width),
                        activation,
                        torch.nn.Linear(width, 1),
                        torch.nn.Sigmoid(),
                    )
                model5 = torch.nn.Sequential(
                        torch.nn.Linear(5, width),
                        activation,
                        torch.nn.Linear(width, width),
                        activation,
                        torch.nn.Linear(width, width),
                        activation,
                        torch.nn.Linear(width, width),
                        activation,
                        torch.nn.Linear(width, 1),
                        torch.nn.Sigmoid(),
                    )
                model9 = torch.nn.Sequential(
                        torch.nn.Linear(5, width),
                        activation,
                        torch.nn.Linear(width, width),
                        activation,
                        torch.nn.Linear(width, width),
                        activation,
                        torch.nn.Linear(width, width),
                        activation,
                        torch.nn.Linear(width, width),
                        activation,
                        torch.nn.Linear(width, width),
                        activation,
                        torch.nn.Linear(width, width),
                        activation,
                        torch.nn.Linear(width, width),
                        activation,
                        torch.nn.Linear(width, 1),
                        torch.nn.Sigmoid(),
                    )
                if depth == 3:
                    models[f'DEPTH {depth}, WIDTH {width}'] = model3
                elif depth == 5:
                    models[f'DEPTH {depth}, WIDTH {width}'] = model5
                elif depth == 9:
                    models[f'DEPTH {depth}, WIDTH {width}'] = model9

        # print(models)
        for config, model in models.items():
            print('')
            print(config)
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters())
            
            def init_weights(m):
                if isinstance(m, torch.nn.Linear):
                    init_type(m.weight)
                    m.bias.data.fill_(0.01)

            model.apply(init_weights)
            model.train()
            for epoch in range(epoch_count):
                for i in range(len(X)):
                    y_hat = model(X.float()[i])
                    loss = loss_fn(y_hat.float(), y.float()[i])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # print(epoch, loss.item())
            with torch.no_grad():
                y_hat = torch.round(model(X.float()))
                train_error = ((y_hat[:, 0] == y).sum() / len(y) * 100).item()
                print('TRAINING ERROR', train_error)
                
                y_hat = torch.round(model(X_test.float()))
                test_error = ((y_hat[:, 0] == y_test).sum() / len(y_test) * 100).item()
                print('TEST ERROR', test_error)

                series = pd.Series({
                    'activation': activation,
                    'Config': config,
                    'Train Error': train_error,
                    'Test Error': test_error
                })
                output_df = output_df.append(series, ignore_index=True).reset_index(drop=True)
    output_df.to_csv('q2e.csv')


def load_bank_note_data(csv: str) -> (pd.DataFrame, pd.Series):
    X_cols = ['WaveletVariance', 'WaveletSkew', 'WaveletCurtosis', 'ImageEntropy']
    y_col = 'Label'

    train = load_data(csv)
    X = train[X_cols]
    return X, train[y_col]


def load_data(csv: str) -> pd.DataFrame:
    return pd.read_csv(
        Path('bank-note', 'bank-note', csv),
        names=['WaveletVariance', 'WaveletSkew', 'WaveletCurtosis', 'ImageEntropy', 'Label']
    )
