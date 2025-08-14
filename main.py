import faulthandler
from pytest import param

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

    if parameters.hparams.train:
        tags = [
            parameters.data_param.dataset_path, #parameters.data_param.dataset_name,
            parameters.data_param.inventory_path, #parameters.data_param.subset,
            parameters.optim_param.optimizer,
            parameters.network_param.network_name,
            f"{'not'*(not parameters.network_param.freeze)} freezed",
            parameters.network_param.pretrained_name,
        ]

        if parameters.hparams.limit_train_batches != 1.0:
            tags += [f"{parameters.hparams.limit_train_batches}_train"]
        if parameters.network_param.freeze_transformer:
            tags += ["transformer_freezed"]

        # wandb.init(
        #     name=run_name,
        #     config=wdb_config,
        #     project=parameters.hparams.wandb_project,
        #     entity=parameters.hparams.wandb_entity,
        #     allow_val_change=True,
        #     job_type="train",
        #     tags=tags,
        # )
        # wandb_run = WandbLogger(
        #     config=wdb_config,
        #     project=parameters.hparams.wandb_project,
        #     entity=parameters.hparams.wandb_entity,
        #     allow_val_change=True,
        # )
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
    else:
        tags = [
            parameters.data_param.dataset_path,  # parameters.data_param.dataset_name,
            parameters.data_param.inventory_path,  # parameters.data_param.subset,
            parameters.network_param.network_name,
            f"{'not'*(not parameters.network_param.freeze)} freezed",
            parameters.network_param.pretrained_name,
            "test",
        ]
        if parameters.hparams.limit_train_batches != 1.0:
            tags += [f"{parameters.hparams.limit_train_batches}_train"]
        if parameters.network_param.freeze_transformer:
            tags += ["transformer_freezed"]

        wandb_run = wandb.init(
            name=parameters.hparams.best_model_run + "_test",
            config=wdb_config,
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
            job_type="test",
            tags=tags,
        )

        wandb_logger = WandbLogger(
            config=wdb_config,
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
        )

        agent = BaseTrainer(parameters, run_name, wandb_logger)
        agent.predict()


if __name__ == "__main__":
    main()
