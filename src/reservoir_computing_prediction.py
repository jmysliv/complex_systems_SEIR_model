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

def test_and_plot(model, X_test, y_test, path, size):
    output = model(X_test)
    # print(output-y_test)
    plt.clf()
    plt.plot(range(size), output[:, 0].view(-1).detach().numpy(), '#0000ffa0',label='Susceptible Predicted')
    plt.plot(range(size), y_test[:, 0].view(-1).detach().numpy(), 'blue',label='Susceptible Original')
    plt.plot(range(size), output[:, 1].view(-1).detach().numpy(), '#ffaa00a0',label='Exposed Predicted')
    plt.plot(range(size), y_test[:, 1].view(-1).detach().numpy(), 'orange',label='Exposed Original')
    plt.plot(range(size), output[:, 2].view(-1).detach().numpy(), '#ff0000a0',label='Infected Predicted')
    plt.plot(range(size), y_test[:, 2].view(-1).detach().numpy(), 'r',label='Infected Original')
    plt.plot(range(size), output[:, 3].view(-1).detach().numpy(), '#000000a0',label='Fatalities Predicted')
    plt.plot(range(size), y_test[:, 3].view(-1).detach().numpy(), 'black',label='Fatalities Original')
    plt.plot(range(size), output[:, 4].view(-1).detach().numpy(), '#00ff00a0',label='Recovered Predicted')
    plt.plot(range(size), y_test[:, 4].view(-1).detach().numpy(), 'g',label='Recovered Original')
    plt.legend()
    plt.savefig(path)
    return output

def test_esn(data, test_size, name, pretraining_data=None):
    mse = []
    max_train_size = int(data.shape[0] * (1 - test_size))
    number_of_points = np.arange(30, max_train_size, 10)
    number_of_layers = [1, 2, 4]

    best_mse = []
    best_mse_pretraining_only = []
    for layers in  number_of_layers:
        mse = []
        mse_pretraining_only = []
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
            # print(X[0].shape,X_test.shape, y.shape, y_test.shape)
            # print(X[0], y[0])
               
            step = int(max_train_size / points_number)
            indices  = torch.arange(0, max_train_size - 1, step)
            X = X[indices]
            y = y[indices]
            
            esn = DeepESN(num_layers=layers, input_size=5)

            if pretraining_data is not None:
                pretraining_sets = tuple([] for _ in range(2))

                for column in pretraining_data:
                    col = pd.DataFrame(data=pretraining_data[column].tolist(), columns=['y'])
                    X_1, X_test_1, y_1, y_test_1 = dl.loader_explicit(col, test_size=1)()

                    pretraining_sets[0].append(torch.cat((X_1,X_test_1)))
                    pretraining_sets[1].append(torch.cat((y_1,y_test_1)))  # I'm not sure how to not split into test and train with the loader_explicit method, so I just do and then join it back together, ugly but works
                
                [X_p, y_p] = [torch.cat((set), dim=1) for set in pretraining_sets ]
                for _ in range(5):
                    esn.fit(X_p, y_p)
                output = test_and_plot(esn, X_test, y_test, f'../output/reservoir/{name}_pretraining_only_layers_{layers}_train_{points_number}',int(data.shape[0] * test_size))
                current_mse = mean_squared_error(y_test, output)
                mse_pretraining_only.append(current_mse)


            esn.fit(X, y)
            output = test_and_plot(esn, X_test, y_test, f'../output/reservoir/{name}_layers_{layers}_train_{points_number}' ,int(data.shape[0] * test_size))
            current_mse = mean_squared_error(y_test, output)
            mse.append(current_mse)
        x_labels = [num/data.shape[0] for num in number_of_points]
        plt.clf()
        plt.plot(x_labels, mse,'b',label='mse')
        plt.xlabel("Relative size of train data")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig(f'../output/reservoir/{name}_layers_{layers}_chart')
        best_mse.append(min(mse))
        if pretraining_data is not None:
            x_labels = [num/data.shape[0] for num in number_of_points]
            plt.clf()
            plt.plot(x_labels, mse_pretraining_only,'b',label='mse')
            plt.xlabel("Relative size of train data")
            plt.ylabel("MSE")
            plt.legend()
            plt.savefig(f'../output/reservoir/{name}_pretraining_only_layers_{layers}_chart')
            best_mse_pretraining_only.append(min(mse))
        
    plt.clf()
    plt.plot(number_of_layers, best_mse,'b',label='mse')
    plt.xlabel("Number of layers")
    plt.ylabel("Best MSE")
    plt.legend()
    plt.savefig(f'../output/reservoir/{name}_layers_chart')

    if pretraining_data is not None:
        plt.clf()
        plt.plot(number_of_layers, best_mse_pretraining_only,'b',label='mse')
        plt.xlabel("Number of groups")
        plt.ylabel("Best MSE")
        plt.legend()
        plt.savefig(f'../output/reservoir/{name}_pretraining_only_layers_chart')



def test_grouped_esn(data, test_size, name, pretraining_data=None):
    mse = []
    max_train_size = int(data.shape[0] * (1 - test_size))
    number_of_points = np.arange(30, max_train_size, 10)
    number_of_groups = [2, 3, 4]

    best_mse = []
    best_mse_pretraining_only = []
    for group in  number_of_groups:
        mse = []
        layers = ((2,)*group)
        mse_pretraining_only = []

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
            step = int(max_train_size / points_number)
            indices  = torch.arange(0, max_train_size - 1, step)
            X = X[indices]
            y = y[indices]
            esn = GroupedDeepESN(num_layers=layers, groups=group, input_size=5)

            if pretraining_data is not None:
                pretraining_sets = tuple([] for _ in range(2))

                for column in pretraining_data:
                    col = pd.DataFrame(data=pretraining_data[column].tolist(), columns=['y'])
                    X_1, X_test_1, y_1, y_test_1 = dl.loader_explicit(col, test_size=1)()

                    pretraining_sets[0].append(torch.cat((X_1,X_test_1)))
                    pretraining_sets[1].append(torch.cat((y_1,y_test_1)))  # I'm not sure how to not split into test and train with the loader_explicit method, so I just do and then join it back together, ugly but works
                
                [X_p, y_p] = [torch.cat((set), dim=1) for set in pretraining_sets ]
                esn.fit(X_p, y_p)
                output = test_and_plot(esn, X_test, y_test, f'../output/reservoir/{name}_pretraining_only_groups_{group}_train_{points_number}',int(data.shape[0] * test_size))
                current_mse = mean_squared_error(y_test, output)
                mse_pretraining_only.append(current_mse)

            esn.fit(X, y)
            output = output = test_and_plot(esn, X_test, y_test, f'../output/reservoir/{name}_groups_{group}_train_{points_number}' ,int(data.shape[0] * test_size))

            current_mse = mean_squared_error(y_test, output)
            mse.append(current_mse)
        x_labels = [num/data.shape[0] for num in number_of_points]
        plt.clf()
        plt.plot(x_labels, mse,'b',label='mse')
        plt.xlabel("Relative size of train data")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig(f'../output/reservoir/{name}_groups_{group}_chart')
        best_mse.append(min(mse))

        if pretraining_data is not None:
            x_labels = [num/data.shape[0] for num in number_of_points]
            plt.clf()
            plt.plot(x_labels, mse_pretraining_only,'b',label='mse')
            plt.xlabel("Relative size of train data")
            plt.ylabel("MSE")
            plt.legend()
            plt.savefig(f'../output/reservoir/{name}_pretraining_only_groups_{group}_chart')
            best_mse_pretraining_only.append(min(mse))

    plt.clf()
    plt.plot(number_of_groups, best_mse,'b',label='mse')
    plt.xlabel("Number of groups")
    plt.ylabel("Best MSE")
    plt.legend()
    plt.savefig(f'../output/reservoir/{name}_groups_chart')

    if pretraining_data is not None:
        plt.clf()
        plt.plot(number_of_groups, best_mse_pretraining_only,'b',label='mse')
        plt.xlabel("Number of groups")
        plt.ylabel("Best MSE")
        plt.legend()
        plt.savefig(f'../output/reservoir/{name}_pretraining_only_groups_chart')


if __name__ == "__main__":
    baseline = pd.read_csv('../output/baseline.csv', usecols=[1, 2, 3, 4, 5])
    test_esn(baseline, 0.8, "esn")
    test_grouped_esn(baseline, 0.8, "grouped_esn")


    assim = pd.read_csv('../output/assimilation/after_assimilation.csv', usecols=[1, 2, 3, 4, 5])
    test_esn(baseline, 0.8, "pretrained_esn", pretraining_data=assim)
    test_grouped_esn(baseline, 0.8, "pretrained_grouped_esn", pretraining_data=assim)