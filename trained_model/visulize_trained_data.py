import sys
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.datasets import load_burgers_1d
from neuralop.training import setup, MGPatchingCallback, SimpleWandBLoggerCallback
from neuralop.utils import get_wandb_api_key, count_model_params
import datetime

# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./burgers_1d_config.yaml", config_name="default", config_folder="../config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

# Set-up distributed communication, if using
device, is_logger = setup(config)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# Set up WandB logging
wandb_init_args = {}
# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Load the Burgers dataset
train_loader, test_loaders = load_burgers_1d(data_path=config.data.folder,
        n_train=config.data.n_train, batch_train=config.data.batch_size, 
        n_test=config.data.n_tests[0], batch_test=config.data.test_batch_sizes[0],
        time=127, grid=[0, 1], device=device
        )

model = get_model(config)

model.load_state_dict(torch.load("/home/qi/Workspace/github/neuraloperator/trained_model/model_240131_1237.pt"))
model = model.to(device)

import matplotlib.pyplot as plt
import numpy as np

test_samples = test_loaders['test'].dataset
# fig = plt.figure(figsize=(15, 12))
# resulution = test_samples.x.shape[2]
# x_axs = np.linspace(0, 1, resulution)
# for index in range(15):
#     # Input x
#     x = test_samples.x[index]
#     # Ground-truth
#     y = test_samples.y[index]
#     # Model prediction
#     out = model(x.to(device).unsqueeze(0))
#     out = out.to('cpu').detach().numpy().reshape((resulution))
#     y = y.reshape((resulution))
#     ax = fig.add_subplot(3, 5, index + 1)
#     ax.plot(x_axs, y, '-')
#     ax.plot(x_axs, out)
#     ax.set_ylim([0, 1])
#     plt.xticks([], [])
#     plt.yticks([], [])

# fig = plt.figure(figsize=(8, 4))
# resulution = test_samples.x.shape[2]
# x_axs = np.linspace(0, 1, resulution)
# for i in range(3):
#     index = i + 0
#     # Input x
#     x = test_samples.x[index]
#     # Ground-truth
#     y = test_samples.y[index]
#     # Model prediction
#     out = model(x.to(device).unsqueeze(0))
#     out = out.to('cpu').detach().numpy().reshape((resulution))
#     y = y.reshape((resulution))
#     ax = fig.add_subplot(1, 3, i + 1)
#     ax.plot(x_axs, y, '-', label='Ground-truth')
#     ax.plot(x_axs, out, label='Prediction')
#     ax.set_ylim([0, 1])
#     ax.set_xlabel('x')
#     if i == 0:
#         ax.legend()
#     plt.xticks([], [])
#     plt.yticks([], [])


# fig.suptitle('Ground-truth output and prediction of FNO.', y=0.98)
# plt.tight_layout()
# plt.show()
# fig.savefig("tmp.png")

x = test_samples.x
out = model(x.to(device))
out = out.to('cpu').detach()
y = test_samples.y

diff = out - y
diff = diff.numpy()

y = y.numpy()

l1_norm = np.linalg.norm(diff, 1, axis=2) / np.linalg.norm(y, 1, axis=2)
l2_norm = np.linalg.norm(diff, 2, axis=2) / np.linalg.norm(y, 2, axis=2)

# l1_norm = np.linalg.norm(diff, 1, axis=2) 
# l2_norm = np.linalg.norm(diff, 2, axis=2) 

print("L1 error: ", l1_norm.mean())
print("L2 error: ", l2_norm.mean())

target_TV = np.sum(np.abs(y[:, :, 1:] - y[:, :, :-1]), axis=2)
predict_TV = np.sum(np.abs(out.numpy()[:, :, 1:] - out.numpy()[:, :, :-1]), axis=2)

rate = predict_TV / target_TV - 1
rate = np.sum(rate)/400

print("Target total variance: ", target_TV.mean())
print("Predict total variance: ", predict_TV.mean())
print("Rate: ", rate)


# Burgers
# tmp1 = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#         0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#         0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#         0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.25, 0.25, 0.25,
#         0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#         0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#         0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#         0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#         0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#         0.25, 0.25]
# tmp1 = np.array(tmp1)

# tmp2 = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75]
# tmp2 = np.array(tmp2)

# LWR
# tmp1 = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75]
# tmp1 = np.array(tmp1)

# tmp2 = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25]
# tmp2 = np.array(tmp2)

# Linear
# tmp1 = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25]
# tmp1 = np.array(tmp1)

# tmp2 = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
#        0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
#        0.75, 0.75]
# tmp2 = np.array(tmp2)

# tmp3 = [0.0000, 0.0099, 0.0198, 0.0297, 0.0396, 0.0495, 0.0594, 0.0693, 0.0792,
#         0.0891, 0.0990, 0.1089, 0.1188, 0.1287, 0.1386, 0.1485, 0.1584, 0.1683,
#         0.1782, 0.1881, 0.1980, 0.2079, 0.2178, 0.2277, 0.2376, 0.2475, 0.2574,
#         0.2673, 0.2772, 0.2871, 0.2970, 0.3069, 0.3168, 0.3267, 0.3366, 0.3465,
#         0.3564, 0.3663, 0.3762, 0.3861, 0.3960, 0.4059, 0.4158, 0.4257, 0.4356,
#         0.4455, 0.4554, 0.4653, 0.4752, 0.4851, 0.4950, 0.5050, 0.5149, 0.5248,
#         0.5347, 0.5446, 0.5545, 0.5644, 0.5743, 0.5842, 0.5941, 0.6040, 0.6139,
#         0.6238, 0.6337, 0.6436, 0.6535, 0.6634, 0.6733, 0.6832, 0.6931, 0.7030,
#         0.7129, 0.7228, 0.7327, 0.7426, 0.7525, 0.7624, 0.7723, 0.7822, 0.7921,
#         0.8020, 0.8119, 0.8218, 0.8317, 0.8416, 0.8515, 0.8614, 0.8713, 0.8812,
#         0.8911, 0.9010, 0.9109, 0.9208, 0.9307, 0.9406, 0.9505, 0.9604, 0.9703,
#         0.9802, 0.9901]
# tmp3 = np.array(tmp3)

# tmp1 = tmp1.reshape((1, 1, 101))
# tmp2 = tmp2.reshape((1, 1, 101))
# tmp3 = tmp3.reshape((1, 1, 101))

# x = np.concatenate((tmp1, tmp3), axis=1)
# x = np.concatenate((tmp2, tmp3), axis=1)
# x = torch.from_numpy(x.astype(np.float32))

# out = model(x.to(device))
# out = out.to('cpu').detach()

# print(out)

