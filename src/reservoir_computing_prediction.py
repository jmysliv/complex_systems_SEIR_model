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


def test_esn(data, test_size, name):
    mse = []
    number_of_points = np.arange(40, 300, 20)
    number_of_layers = [1, 2, 3]

    for layers in  number_of_layers:
        mse = []
        for points_number in number_of_points:
            seidr = dl.loader_explicit(data, test_size=int(data.size * test_size))
            X, X_test, y, y_test = seidr()
            step = int(data.size * (1 - test_size) / number_of_points)
            indices  = torch.arange(0, int(data.size * (1 - test_size)), step)
            X = X[indices]
            y = y[indices]
            esn = DeepESN(num_layers=layers)
            esn.fit(X, y)
            output = esn(X_test)
            plt.clf()
            plt.plot(range(int(data.size * test_size)), output.view(-1).detach().numpy(), 'r',label='predicted')
            plt.plot(range(int(data.size * test_size)), y_test.view(-1).detach().numpy(), 'b',label='original')
            plt.legend()
            plt.savefig(f'../output/reservoir/{name}_layers_{layers}_train_{points_number}')
            current_mse = mean_squared_error(y_test, output)
            mse.append(current_mse)
        plt.plot(number_of_points, mse,'b',label='mse')
        plt.xlabel("Train size")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig(f'../output/reservoir/{name}_layers_{layers}_chart')
        



def test_grouped_esn(data, test_size, name):
    mse = []
    number_of_points = np.arange(40, 300, 20)
    number_of_groups = [2, 3, 4]

    for group in  number_of_groups:
        mse = []
        layers = ((2,)*group)
        for points_number in number_of_points:
            seidr = dl.loader_explicit(data, test_size=int(data.size * test_size))
            X, X_test, y, y_test = seidr()
            step = int(data.size * (1 - test_size) / number_of_points)
            indices  = torch.arange(0, int(data.size * (1 - test_size)), step)
            X = X[indices]
            y = y[indices]
            esn = GroupedDeepESN(num_layers=layers, groups=group)
            esn.fit(X, y)
            output = esn(X_test)
            plt.clf()
            plt.plot(range(int(data.size * test_size)), output.view(-1).detach().numpy(), 'r',label='predicted')
            plt.plot(range(int(data.size * test_size)), y_test.view(-1).detach().numpy(), 'b',label='original')
            plt.legend()
            plt.savefig(f'../output/reservoir/{name}_groups_{group}_train_{points_number}')
            current_mse = mean_squared_error(y_test, output)
            mse.append(current_mse)
        plt.plot(number_of_points, mse,'b',label='mse')
        plt.xlabel("Train size")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig(f'../output/reservoir/{name}_groups_{group}_chart')

if __name__ == "__main__":
    baseline = pd.read_csv('../output/baseline.csv')
    susceptible = pd.DataFrame(data=baseline['Susceptible'].tolist(), columns=['y'])
    test_esn(susceptible, 0.2, "esn")
    test_grouped_esn(susceptible, 0.2, "grouped_esn")
