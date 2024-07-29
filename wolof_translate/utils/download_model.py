import shutil
import wandb
import glob
import os


def transfer_model(artifact_dir: str, model_name: str):
    """Transfer a download artifact into another directory

    Args:
        artifact_dir (str): The directory of the artifact
        model_name (str): The name of the model
    """
    # transfer the model inside the artifact to data/checkpoints/name_of_model
    os.makedirs(model_name, exist_ok=True)
    for file in glob.glob(f"{artifact_dir}/*"):
        shutil.copy(file, model_name)

    # delete the artifact
    shutil.rmtree(artifact_dir)


def download_artifact(artifact_name: str, model_name: str, type_: str = "dataset"):
    """This function download an artifact from weights and bias and store it into a directory

    Args:
        artifact_name (str): name of the artifact
        model_name (str): name of the model
        type (str): type of the artifact. Default to 'directory'.
    """
    # download wandb model
    run = wandb.init()
    artifact = run.use_artifact(artifact_name, type=type_)
    artifact_dir = artifact.download()

    # transfer the artifact into another directory
    transfer_model(artifact_dir, model_name)

    # finish wandb
    wandb.finish()
