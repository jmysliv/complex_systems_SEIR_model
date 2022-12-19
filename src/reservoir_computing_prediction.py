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
    number_of_layers = [1, 2, 4]

    for layers in  number_of_layers:
        mse = []
        for points_number in number_of_points:
            sets = tuple([] for _ in range(4))
            for column in data:
                col = pd.DataFrame(data=data[column].tolist(), columns=['y'])
                X_1, X_test_1, y_1, y_test_1 = dl.loader_explicit(col, test_size=int(data.shape[0] * test_size))()
                sets[0].append(X_1)
                sets[1].append(X_test_1)
                sets[2].append(y_1)
                sets[3].append(y_test_1)
            
            [X, X_test, y, y_test] = [torch.cat((set), dim=1) for set in sets ]

               
            step = int(data.shape[0] * (1 - test_size) / points_number)
            indices  = torch.arange(0, int(data.shape[0] * (1 - test_size)) - 1, step)
            X = X[indices]
            y = y[indices]
            esn = DeepESN(num_layers=layers, input_size=5)
            esn.fit(X, y)
            output = esn(X_test)
            plt.clf()
            plt.plot(range(int(data.shape[0] * test_size)), output[:, 0].view(-1).detach().numpy(), '#0000ffa0',label='Susceptible Predicted')
            plt.plot(range(int(data.shape[0] * test_size)), y_test[:, 0].view(-1).detach().numpy(), 'blue',label='Susceptible Original')
            plt.plot(range(int(data.shape[0] * test_size)), output[:, 1].view(-1).detach().numpy(), '#ffaa00a0',label='Exposed Predicted')
            plt.plot(range(int(data.shape[0] * test_size)), y_test[:, 1].view(-1).detach().numpy(), 'orange',label='Exposed Original')
            plt.plot(range(int(data.shape[0] * test_size)), output[:, 2].view(-1).detach().numpy(), '#ff0000a0',label='Infected Predicted')
            plt.plot(range(int(data.shape[0] * test_size)), y_test[:, 2].view(-1).detach().numpy(), 'r',label='Infected Original')
            plt.plot(range(int(data.shape[0] * test_size)), output[:, 3].view(-1).detach().numpy(), '#000000a0',label='Fatalities Predicted')
            plt.plot(range(int(data.shape[0] * test_size)), y_test[:, 3].view(-1).detach().numpy(), 'black',label='Fatalities Original')
            plt.plot(range(int(data.shape[0] * test_size)), output[:, 4].view(-1).detach().numpy(), '#00ff00a0',label='Recovered Predicted')
            plt.plot(range(int(data.shape[0] * test_size)), y_test[:, 4].view(-1).detach().numpy(), 'g',label='Recovered Original')
            plt.legend()
            plt.savefig(f'../output/reservoir/{name}_layers_{layers}_train_{points_number}')
            current_mse = mean_squared_error(y_test, output)
            mse.append(current_mse)
        x_labels = [num/data.shape[0] for num in number_of_points]
        plt.clf()
        plt.plot(x_labels, mse,'b',label='mse')
        plt.xlabel("Relative size of train data")
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
            sets = tuple([] for _ in range(4))
            for column in data:
                col = pd.DataFrame(data=data[column].tolist(), columns=['y'])
                X_1, X_test_1, y_1, y_test_1 = dl.loader_explicit(col, test_size=int(data.shape[0] * test_size))()
                sets[0].append(X_1)
                sets[1].append(X_test_1)
                sets[2].append(y_1)
                sets[3].append(y_test_1)
            
            [X, X_test, y, y_test] = [torch.cat((set), dim=1) for set in sets ]

            step = int(data.shape[0] * (1 - test_size) / points_number)
            indices  = torch.arange(0, int(data.shape[0] * (1 - test_size)) - 1, step)
            X = X[indices]
            y = y[indices]
            esn = GroupedDeepESN(num_layers=layers, groups=group, input_size=5)
            esn.fit(X, y)
            output = esn(X_test)
            plt.clf()
            plt.plot(range(int(data.shape[0] * test_size)), output[:, 0].view(-1).detach().numpy(), '#0000ffa0',label='Susceptible Predicted')
            plt.plot(range(int(data.shape[0] * test_size)), y_test[:, 0].view(-1).detach().numpy(), 'blue',label='Susceptible Original')
            plt.plot(range(int(data.shape[0] * test_size)), output[:, 1].view(-1).detach().numpy(), '#ffaa00a0',label='Exposed Predicted')
            plt.plot(range(int(data.shape[0] * test_size)), y_test[:, 1].view(-1).detach().numpy(), 'orange',label='Exposed Original')
            plt.plot(range(int(data.shape[0] * test_size)), output[:, 2].view(-1).detach().numpy(), '#ff0000a0',label='Infected Predicted')
            plt.plot(range(int(data.shape[0] * test_size)), y_test[:, 2].view(-1).detach().numpy(), 'r',label='Infected Original')
            plt.plot(range(int(data.shape[0] * test_size)), output[:, 3].view(-1).detach().numpy(), '#000000a0',label='Fatalities Predicted')
            plt.plot(range(int(data.shape[0] * test_size)), y_test[:, 3].view(-1).detach().numpy(), 'black',label='Fatalities Original')
            plt.plot(range(int(data.shape[0] * test_size)), output[:, 4].view(-1).detach().numpy(), '#00ff00a0',label='Recovered Predicted')
            plt.plot(range(int(data.shape[0] * test_size)), y_test[:, 4].view(-1).detach().numpy(), 'g',label='Recovered Original')
            plt.legend()
            plt.savefig(f'../output/reservoir/{name}_groups_{group}_train_{points_number}')
            current_mse = mean_squared_error(y_test, output)
            mse.append(current_mse)
        x_labels = [num/data.shape[0] for num in number_of_points]
        plt.clf()
        plt.plot(x_labels, mse,'b',label='mse')
        plt.xlabel("Relative size of train data")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig(f'../output/reservoir/{name}_groups_{group}_chart')

if __name__ == "__main__":
    baseline = pd.read_csv('../output/baseline.csv', usecols=[1, 2, 3, 4, 5])
    test_esn(baseline, 0.2, "esn")
    test_grouped_esn(baseline, 0.2, "grouped_esn")
