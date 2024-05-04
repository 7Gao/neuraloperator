from pathlib import Path
import torch
import numpy as np
from .tensor_dataset import TensorDataset
import scipy

def load_burgers_1d(
    data_path, n_train, n_test, batch_train=32, batch_test=100, time=1, grid=[0, 1],
    device=torch.device('cpu')
):

    # data_path = Path(data_path).as_posix()
    # data = torch.load(data_path)
    data = scipy.io.loadmat(data_path)
    x_train = data['init_Y'][0:n_train,:].astype(np.float32)
    x_train = torch.from_numpy(x_train)

    x_test = data['init_Y'][n_train : (n_train + n_test), :].astype(np.float32)
    x_test = torch.from_numpy(x_test)

    y_train = data['term_Y'][0:n_train, :].astype(np.float32)
    y_train = torch.from_numpy(y_train)

    y_test = data['term_Y'][n_train : (n_train + n_test), :].astype(np.float32)
    y_test = torch.from_numpy(y_test)

    s = x_train.size(-1)

    if grid is not None:
        grid = torch.linspace(grid[0], grid[1], s + 1)[0:-1].view(1, -1)

        grid_train = grid.repeat(n_train, 1)
        grid_test = grid.repeat(n_test, 1)

        x_train = torch.cat((x_train.unsqueeze(1), grid_train.unsqueeze(1)), 1)
        x_test = torch.cat((x_test.unsqueeze(1), grid_test.unsqueeze(1)), 1)
        y_train = y_train.unsqueeze(1)
        y_test = y_test.unsqueeze(1)

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

# def load_burgers_1d(
#     data_path, n_train, n_test, batch_train=32, batch_test=100, time=1, grid=[0, 1],
#     device=torch.device('cpu')
# ):

#     # data_path = Path(data_path).as_posix()
#     # data = torch.load(data_path)
#     data = torch.from_numpy(scipy.io.loadmat(data_path)['output'].astype(np.float32))
#     x_train = data[0:n_train, 0, :]
#     x_test = data[n_train : (n_train + n_test), 0, :]

#     y_train = data[0:n_train, time, :]
#     y_test = data[n_train : (n_train + n_test), time, :]

#     s = x_train.size(-1)

#     if grid is not None:
#         grid = torch.linspace(grid[0], grid[1], s + 1)[0:-1].view(1, -1)

#         grid_train = grid.repeat(n_train, 1)
#         grid_test = grid.repeat(n_test, 1)

#         x_train = torch.cat((x_train.unsqueeze(1), grid_train.unsqueeze(1)), 1)
#         x_test = torch.cat((x_test.unsqueeze(1), grid_test.unsqueeze(1)), 1)
#         y_train = y_train.unsqueeze(1)
#         y_test = y_test.unsqueeze(1)

#     x_train = x_train.to(device)
#     x_test = x_test.to(device)
#     y_train = y_train.to(device)
#     y_test = y_test.to(device)

#     train_loader = torch.utils.data.DataLoader(
#         TensorDataset(x_train, y_train),
#         batch_size=batch_train,
#         shuffle=False,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         TensorDataset(x_test, y_test),
#         batch_size=batch_test,
#         shuffle=False,
#     )

#     test_loaders = {'test':test_loader}

#     return train_loader, test_loaders

def load_burgers_1dtime(
        data_path, n_train, n_test, batch_size=32, batch_size_test=100, 
        temporal_length=101, spatial_length=128, temporal_subsample=1, 
        spatial_subsample=1, pad=0, device=torch.device('cpu')):
    """
    Load burgers.mat data. Given the initial condition (t=0),
    predict timesteps 1 to temporal_length.
    """
    # with np.load(data_path) as data:
    #     x_data = data['input']
    #     y_data = data['output']
    #     visc = data['visc']

    data = scipy.io.loadmat(data_path)
    x_data = data['input']
    y_data = data['output']
    visc = data['visc']

    x_data = torch.from_numpy(x_data.astype(np.float32))
    x_data = x_data[:, :spatial_length:spatial_subsample]
    y_data = torch.from_numpy(y_data.astype(np.float32))
    y_data = y_data[:, :temporal_length:temporal_subsample, :spatial_length:spatial_subsample]
    visc = torch.from_numpy(visc.astype(np.float32)).item()

    x_train = x_data[:n_train]
    y_train = y_data[:n_train]
    x_test = x_data[n_train:n_train+n_test]
    y_test = y_data[n_train:n_train+n_test]

    domain_lengths = [spatial_length / 128, (temporal_length - 1) / 100]
    domain_starts = [0., 0.]

    spatial_length = spatial_length // spatial_subsample
    temporal_length = temporal_length // temporal_subsample

    if pad:
        x_train = torch.nn.ReplicationPad1d(pad)(x_train)
        x_test = torch.nn.ReplicationPad1d(pad)(x_test)
        spatial_length += 2 * pad
        temporal_length += 2 * pad
        incrs = [spatial_subsample / 128, temporal_subsample / 100]
        domain_lengths = [d + incr * pad for d, incr in zip(domain_lengths, incrs)]
        domain_starts = [-incr * pad for incr in incrs]

    # TODO: use include_endpoint arg here
    grid_x = torch.tensor(np.linspace(domain_starts[0], domain_lengths[0], spatial_length + 1)[:-1], dtype=torch.float)
    grid_t = torch.tensor(np.linspace(domain_starts[1], domain_lengths[1], temporal_length), dtype=torch.float)

    grid_x = grid_x.reshape(1, 1, spatial_length)
    grid_t = grid_t.reshape(1, temporal_length, 1)

    x_train = x_train.reshape(n_train, 1, spatial_length).repeat([1, temporal_length, 1])
    x_test = x_test.reshape(n_test, 1, spatial_length).repeat([1, temporal_length, 1])

    # TODO: add option to not have positional encoding
    x_train = torch.stack([x_train, 
                           grid_t.repeat([n_train, 1, spatial_length]),
                           grid_x.repeat([n_train, temporal_length, 1]) 
                           ], dim=3)
    x_test = torch.stack([x_test, 
                          grid_t.repeat([n_test, 1, spatial_length]),
                          grid_x.repeat([n_test, temporal_length, 1]) 
                          ], dim=3)

    x_train = x_train.permute(0, 3, 1, 2)
    x_test = x_test.permute(0, 3, 1, 2)
    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=False)

    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_db, batch_size=batch_size_test, shuffle=False)

    output_encoder = None
    test_loaders = {'test':test_loader}

    return train_loader, test_loaders, output_encoder