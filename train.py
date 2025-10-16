import faulthandler
from pytest import param
from pathlib import Path
faulthandler.enable()

from pytorch_lightning.loggers import WandbLogger

# Standard libraries
import wandb
from agents.BaseTrainer import BaseTrainer
from config.hparams import Parameters
from utils.agent_utils import parse_params, get_run_name


def main():
    parameters = Parameters.parse()

    # initialize wandb instance
    wdb_config = parse_params(parameters)
    run_name = get_run_name(parameters)
    tags = [
        Path(parameters.data_param.dataset_path).name,
        Path(parameters.data_param.inventory_path).name,
        parameters.optim_param.optimizer,
        parameters.network_param.network_name,
        f"{'not'*(not parameters.network_param.freeze)} freezed",
        parameters.network_param.pretrained_name,
    ]

    if parameters.hparams.limit_train_batches != 1.0:
        tags += [f"{parameters.hparams.limit_train_batches}_train"]
    if parameters.network_param.freeze_transformer:
        tags += ["transformer_freezed"]

    run = wandb.init(
        id=run_name,  # This is the key - use hash as ID
        project=parameters.hparams.wandb_project,
        entity=parameters.hparams.wandb_entity,
        config=wdb_config,
        job_type="train",
        tags=tags,
        resume="allow",  # Allow resuming if ID exists
        name=run_name  # Display name
    )

    # Now create WandbLogger with the existing run
    wandb_logger = WandbLogger(experiment=run)
    agent = BaseTrainer(parameters, run_name, wandb_logger)
    agent.run()


if __name__ == "__main__":
    main()
