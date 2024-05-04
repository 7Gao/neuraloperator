import sys
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.datasets import load_ngsim, load_ngsim_1d
from neuralop.training import setup, MGPatchingCallback, SimpleWandBLoggerCallback
from neuralop.utils import get_wandb_api_key, count_model_params
import datetime
import numpy as np

def main(data_name='NGSIM_sikp3_gap42', train_size=400, test_size=176, include_rho= False,
    include_speed=False, include_upstream=False, include_downstream=False):
    # Read the configuration
    config_name = "default"
    pipe = ConfigPipeline(
        [
            YamlConfig(
                "./ngsim_1d_config.yaml", config_name="default", config_folder="../config"
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder="../config"),
        ]
    )
    config = pipe.read_conf()
    config_name = pipe.steps[-1].config_name

    # Set-up distributed communication, if using
    device, is_logger = setup(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # Set up WandB logging
    wandb_init_args = {}
    # Make sure we only print information when needed
    config.verbose = config.verbose and is_logger

    # Print config to screen
    if config.verbose:
        pipe.log()
        sys.stdout.flush()

    # Load the Burgers dataset
    # data_path = '/home/qi/Workspace/github/neuraloperator/datas/ngsim/NGSIM_skip3_gap42.npz'
    data_path = '/home/qi/Workspace/github/neuraloperator/datas/ngsim/'+data_name+'.npz'
    train_loader, test_loaders = load_ngsim_1d(data_path=data_path,
            n_train=train_size, batch_train=config.data.batch_size, 
            n_test=test_size, batch_test=config.data.test_batch_sizes[0],
            grid=[0, 1], include_rho = include_rho, include_speed=include_speed, 
            include_upstream=include_upstream,include_downstream=include_downstream, 
            device=device)

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
    if config.opt.training_loss == "l2":
        train_loss = l2loss
    elif config.opt.training_loss == "h1":
        train_loss = h1loss
    else:
        raise ValueError(
            f'Got training_loss={config.opt.training_loss} '
            f'but expected one of ["l2", "h1"]'
        )
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
            encoder=None
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

    result_path =  '/home/qi/Workspace/github/neuraloperator/results/ngsim/FNO_pred_'+data_name+'.npz'

    pred = model(test_loaders['test'].dataset.x.to(device))
    pred = pred.cpu().detach().numpy()
    true = test_loaders['test'].dataset.y.cpu().detach().numpy()

    np.savez(result_path, y_pred=pred, y_true=true)


if __name__ == "__main__":
    # main(data_name='NGSIM_rhou_ud_skip3_gap42', train_size=400, test_size=176, 
    #     include_speed=False, include_upstream=True, include_downstream=True)
    
    # main(data_name='NGSIM_rhou_ud_skip1_gap42', train_size=1500, test_size=229, 
    #     include_rho = False, include_speed=True, include_upstream=False, 
    #     include_downstream=False)
    
    main(data_name='NGSIM_rhou_ud_skip3_gap42', train_size=400, test_size=176,
        include_rho = True, include_speed=True, include_upstream=True, 
        include_downstream=True)



