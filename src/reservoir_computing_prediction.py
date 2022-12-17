import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import auto_esn.utils.dataset_loader as dl
from auto_esn.datasets.df import MackeyGlass
from auto_esn.esn.esn import GroupedDeepESN,DeepESN
from auto_esn.esn.reservoir.util import NRMSELoss
from auto_esn.esn.reservoir.activation import tanh


def compare_mse(data, test_size):
    mse = []
    number_of_points = np.arange(40, 300, 20)

    for points_number in number_of_points:
        seidr = dl.loader_explicit(data, test_size=int(data.size * test_size))
        X, X_test, y, y_test = seidr()
        step = int(data.size * (1 - test_size) / number_of_points)
        indices  = torch.arange(0, int(data.size * (1 - test_size)), step)
        X = X[indices]
        y = y[indices]
        esn = DeepESN(num_layers=1)
        esn.fit(X, y)
        output = esn(X_test)
        plt.clf()
        plt.plot(range(int(data.size * test_size)), output.view(-1).detach().numpy(), 'r',label='predicted')
        plt.plot(range(int(data.size * test_size)), y_test.view(-1).detach().numpy(), 'b',label='original')
        plt.legend()
        plt.save(f'../output/reservoir/prediction_{points_number}')
        current_mse = mean_squared_error(y_test, output)
        mse.append(current_mse)
    
    plt.plot(number_of_points, mse,'b',label='mse')
    plt.xlabel("Train size")
    plt.ylabel("MSE")
    plt.legend()
    plt.save(f'../output/reservoir/chart_reservoir')

if __name__ == "__main__":
    baseline = pd.read_csv('../output/baseline.csv')
    susceptible = pd.DataFrame(data=baseline['Susceptible'].tolist(), columns=['y'])
    compare_mse(susceptible)
