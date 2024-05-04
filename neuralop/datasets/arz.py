from pathlib import Path
import torch
import numpy as np
from .tensor_dataset import TensorDataset
import scipy

def load_arz_1d(
    data_path, 
    n_train, 
    n_test, 
    batch_train=32, 
    batch_test=100, 
    grid=[0, 1],
    device=torch.device('cpu')
):
    data = np.load(data_path)

    u_inits = data['u_inits'].astype(np.float32)
    rho_inits = data['rho_inits'].astype(np.float32)
    u_results = data['u_results'].astype(np.float32)
    rho_results = data['rho_results'].astype(np.float32)

    u_inits = torch.from_numpy(u_inits)
    rho_inits = torch.from_numpy(rho_inits)
    u_results = torch.from_numpy(u_results)
    rho_results = torch.from_numpy(rho_results)

    u_inits = u_inits.unsqueeze(1)
    rho_inits = rho_inits.unsqueeze(1)
    u_results = u_results.unsqueeze(1)
    rho_results = rho_results.unsqueeze(1)

    x_train = torch.cat((u_inits[0:n_train, :, :], rho_inits[0:n_train, :, :]), 1)
    x_test = torch.cat((u_inits[n_train:n_train+n_test, :, :], rho_inits[n_train:n_train+n_test, :, :]), 1)
    y_train = torch.cat((u_results[0:n_train, :, :], rho_results[0:n_train, :, :]), 1)
    y_test = torch.cat((u_results[n_train:n_train+n_test, :, :], rho_results[n_train:n_train+n_test, :, :]), 1)

    s = u_inits.size(-1)

    if grid is not None:
        grid = torch.linspace(grid[0], grid[1], s + 1)[0:-1].view(1, -1)

        grid_train = grid.repeat(n_train, 1)
        grid_test = grid.repeat(n_test, 1)

        x_train = torch.cat((x_train, grid_train.unsqueeze(1)), 1)
        x_test = torch.cat((x_test, grid_test.unsqueeze(1)), 1)
        y_train = y_train
        y_test = y_test

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_train,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_test,
        shuffle=False,
    )

    test_loaders = {'test':test_loader}

    return train_loader, test_loaders
