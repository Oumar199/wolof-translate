import wandb


def add_directory(
    directory: str,
    artifact_name: str,
    project: str = "fw_artifacts",
    entity: str = "oumar-kane-team",
):
    """Initialize a project and add checkpoints as artifact to wandb

    Args:
        directory (str): The directory where are stored the checkpoints
        artifact_name (_type_): The name of the artifact
        project (str, optional): The project name. Defaults to 'fw_artifacts'.
        entity (str, optional): The entity name. Defaults to 'oumar-kane-team'.
    """

    run = wandb.init(project=project, entity=entity)

    # add a directory as artifact to wandb
    artifact = wandb.Artifact(artifact_name, type="dataset")
    artifact.add_dir(directory)
    run.log_artifact(artifact)

    wandb.finish()
