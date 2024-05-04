from pathlib import Path
import torch
import numpy as np
from .tensor_dataset import TensorDataset
import scipy

def load_ngsim_1d(
    data_path, 
    n_train, 
    n_test, 
    batch_train=32, 
    batch_test=100, 
    grid=[0, 1],
    include_rho=True,
    include_speed=False,
    include_upstream=False,
    include_downstream=False,
    device=torch.device('cpu')
):
    data = np.load(data_path)

    inits = []
    results = []
    if include_rho:
        rho_inits = data['rho_inits'].astype(np.float32)
        rho_results = data['rho_results'].astype(np.float32)

        inits.append(rho_inits)
        results.append(rho_results)

        if include_upstream:
            rho_upstream = data['rho_upstream'].astype(np.float32)
            inits.append(rho_upstream)

        if include_downstream:
            rho_downstream = data['rho_downstream'].astype(np.float32)
            inits.append(rho_downstream)

    if include_speed:
        u_inits = data['u_inits'].astype(np.float32)
        u_results = data['u_results'].astype(np.float32)

        inits.append(u_inits)
        results.append(u_results)

        if include_upstream:
            u_upstream = data['u_upstream'].astype(np.float32)
            inits.append(u_upstream)

        if include_downstream:
            u_downstream = data['u_downstream'].astype(np.float32)
            inits.append(u_downstream)

    inits = [torch.from_numpy(init) for init in inits]
    inits = [init.unsqueeze(1) for init in inits]
    

    results = [torch.from_numpy(result) for result in results]  
    results = [result.unsqueeze(1) for result in results]

    x = torch.cat(inits, 1)
    y = torch.cat(results, 1)


    print(x.shape)

    x_train = x[0:n_train, :, :]
    x_test = x[n_train:n_train+n_test, :, :]
    y_train = y[0:n_train, :, :]
    y_test = y[n_train:n_train+n_test, :, :]

    s = x.size(-1)

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


def load_ngsim_1dtime(
    data_path, 
    n_train, 
    n_test, 
    batch_train=32, 
    batch_test=100, 
    temporal_length=101, 
    spatial_length=128,
    use_density=True,
    use_speed=True,
    use_upstream=True,
    use_downstream=True,
    device=torch.device('cpu')
):
    data = np.load(data_path)

    rho_xt_inits = data['rho_xt_inits'].astype(np.float32)
    rho_xt_results = data['rho_xt_results'].astype(np.float32)
    u_xt_inits = data['u_xt_inits'].astype(np.float32)
    u_xt_results = data['u_xt_results'].astype(np.float32)
    
    # normalize
    max_rho = np.max(rho_xt_results)
    max_u = np.max(u_xt_results)
    rho_xt_inits = rho_xt_inits / max_rho
    rho_xt_results = rho_xt_results / max_rho
    u_xt_inits = u_xt_inits / max_u
    u_xt_results = u_xt_results / max_u

    rho_xt_inits = torch.from_numpy(rho_xt_inits)
    rho_xt_results = torch.from_numpy(rho_xt_results)
    u_xt_inits = torch.from_numpy(u_xt_inits)
    u_xt_results = torch.from_numpy(u_xt_results)

    in_channels = 2
    out_channels = 0

    x_data = []
    y_data = []

    if use_density:
        in_channels += 1
        out_channels += 1
        x_data.append(rho_xt_inits[:, 0, :, :].unsqueeze(1))
        y_data.append(rho_xt_results.unsqueeze(1))

        if use_upstream:
            in_channels += 1
            x_data.append(rho_xt_inits[:, 1, :, :].unsqueeze(1))
        if use_downstream:
            in_channels += 1
            x_data.append(rho_xt_inits[:, 2, :, :].unsqueeze(1))

    if use_speed:
        in_channels += 1
        out_channels += 1
        x_data.append(u_xt_inits[:, 0, :, :].unsqueeze(1))
        y_data.append(u_xt_results.unsqueeze(1))

        if use_upstream:
            in_channels += 1
            x_data.append(u_xt_inits[:, 1, :, :].unsqueeze(1))
        if use_downstream:
            in_channels += 1
            x_data.append(u_xt_inits[:, 2, :, :].unsqueeze(1))

    if out_channels == 0:
        raise ValueError('No input channels specified')

    x_data = torch.cat(x_data, 1)
    y_data = torch.cat(y_data, 1)

    x_train = x_data[:n_train]
    y_train = y_data[:n_train]
    x_test = x_data[n_train:n_train+n_test]
    y_test = y_data[n_train:n_train+n_test]

    grid_x = torch.tensor(np.linspace(0, 1, spatial_length + 1)[:-1], dtype=torch.float)
    grid_t = torch.tensor(np.linspace(0, 1, temporal_length), dtype=torch.float)

    grid_x = grid_x.reshape(1, 1, 1, spatial_length).repeat([1, 1, temporal_length, 1])
    grid_t = grid_t.reshape(1, 1, temporal_length, 1).repeat([1, 1, 1, spatial_length])
    
    x_train = torch.cat([x_train, 
                           grid_t.repeat([n_train, 1, 1, 1]),
                           grid_x.repeat([n_train, 1, 1, 1]) 
                           ], dim=1)
    x_test = torch.cat([x_test, 
                          grid_t.repeat([n_test, 1, 1, 1]),
                          grid_x.repeat([n_test, 1, 1, 1]) 
                          ], dim=1)



    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_train, shuffle=False)

    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_db, batch_size=batch_test, shuffle=False)

    output_encoder = None
    test_loaders = {'test':test_loader}

    return train_loader, test_loaders, output_encoder, (in_channels, out_channels)

