import sys
import torch
import wandb
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from neuralop import H1Loss, LpLoss, BurgersEqnLoss, ICLoss, WeightedSumLoss, Trainer, get_model
from neuralop.datasets import load_burgers_1dtime
from neuralop.training import setup, MGPatchingCallback, SimpleWandBLoggerCallback
from neuralop.utils import get_wandb_api_key, count_model_params


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./burgers_config.yaml", config_name="default", config_folder="../config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

# Set-up distributed communication, if using
device, is_logger = setup(config)

# Set up WandB logging
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.fno2d.n_layers,
                config.fno2d.n_modes_width,
                config.fno2d.n_modes_height,
                config.fno2d.hidden_channels,
                config.fno2d.factorization,
                config.fno2d.rank,
                config.patching.levels,
                config.patching.padding,
            ]
        )
    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]

else: 
    wandb_init_args = {}
# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose:
    pipe.log()
    sys.stdout.flush()

# Load the Burgers dataset
train_loader, test_loaders, output_encoder = load_burgers_1dtime(data_path=config.data.folder,
        n_train=config.data.n_train, batch_size=config.data.batch_size, 
        n_test=config.data.n_tests[0], batch_size_test=config.data.test_batch_sizes[0],
        temporal_length=config.data.temporal_length, spatial_length=config.data.spatial_length,
        pad=config.data.get("pad", 0), temporal_subsample=config.data.get("temporal_subsample", 1),
        spatial_subsample=config.data.get("spatial_subsample", 1),
        )

model = get_model(config)
model = model.to(device)

# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    )

# Create the optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")


# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
ic_loss = ICLoss()
equation_loss = BurgersEqnLoss(method=config.opt.get('pino_method', None), 
                               visc=0.01, loss=F.mse_loss)

training_loss = config.opt.training_loss
if not isinstance(training_loss, (tuple, list)):
    training_loss = [training_loss]

losses = []
weights = []
for loss in training_loss:
    # Append loss
    if loss == 'l2':
        losses.append(l2loss)
    elif loss == 'h1':
        losses.append(h1loss)
    elif loss == 'equation':
        losses.append(equation_loss)
    elif loss == 'ic':
        losses.append(ic_loss)
    else:
        raise ValueError(f'Training_loss={loss} is not supported.')

    # Append loss weight
    if "loss_weights" in config.opt:
        weights.append(config.opt.loss_weights.get(loss, 1.))
    else:
        weights.append(1.)

train_loss = WeightedSumLoss(losses=losses, weights=weights)
eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

# only perform MG patching if config patching levels > 0

callbacks = [
    MGPatchingCallback(
        levels=config.patching.levels,
        padding_fraction=config.patching.padding,
        stitching=config.patching.stitching, 
        encoder=output_encoder
    ),
    SimpleWandBLoggerCallback(**wandb_init_args)
]


trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    device=device,
    amp_autocast=config.opt.amp_autocast,
    callbacks=callbacks,
    log_test_interval=config.wandb.log_test_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
    wandb_log = config.wandb.log
)

# Log parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log)
        wandb.watch(model)


trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

if config.wandb.log and is_logger:
    wandb.finish()


import matplotlib.pyplot as plt
import numpy as np

test_samples = test_loaders['test'].dataset
fig = plt.figure(figsize=(15, 9))
x_axs = np.linspace(0, 1, 128)
for index in range(3):
    # Input x
    x = test_samples.x[index]
    # Ground-truth
    y = test_samples.y[index]
    # Model prediction
    out = model(x.unsqueeze(0))

    y = y.reshape((128, 128))
    out = out.detach().numpy().reshape((128, 128))
    ax = fig.add_subplot(3, 5, index*5 + 1)
    ax.plot(x_axs, y[0], '-')
    ax.plot(x_axs, out[0])
    ax.set_ylim([0, 1])
    if index == 0: 
        ax.set_title('t = 0')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 2)
    ax.plot(x_axs, y[31], '-')
    ax.plot(x_axs, out[31])
    ax.set_ylim([0, 1])
    if index == 0: 
        ax.set_title('t = 0.25')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 3)
    ax.plot(x_axs, y[63], '-')
    ax.plot(x_axs, out[63])
    ax.set_ylim([0, 1])
    if index == 0: 
        ax.set_title('t = 0.5')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 4)
    ax.plot(x_axs, y[95], '-')
    ax.plot(x_axs, out[95])
    ax.set_ylim([0, 1])
    if index == 0: 
        ax.set_title('t = 0.75')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 5)
    ax.plot(x_axs, y[127], '-')
    ax.plot(x_axs, out[127])
    ax.set_ylim([0, 1])
    if index == 0: 
        ax.set_title('t = 1')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.savefig("burgers implicit 1 shock slice example.png")

test_samples = test_loaders['test'].dataset
fig = plt.figure(figsize=(7, 7))
for index in range(3):
    # Input x
    x = test_samples.x[index]
    # Ground-truth
    y = test_samples.y[index]
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 2, index*2 + 1)
    ax.imshow(y.squeeze(), vmin = 0, vmax = 1)
    if index == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 2, index*2+ 2)
    ax.imshow(out.squeeze().detach().numpy(), vmax = 1, vmin = 0)
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.savefig("burgers implicit 1 shock example.png")