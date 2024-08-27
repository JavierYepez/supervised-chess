from pathlib import Path
from typing import Callable, Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.tensorboard.writer import SummaryWriter
from torch._prims_common import DeviceLikeType
from tqdm.auto import tqdm

from .early_stop import EarlyStop
from ..settings import Settings


def train_one_epoch(
    training_loader: DataLoader,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    epoch_index: int,
    device: DeviceLikeType,
    tb_writer: SummaryWriter,
):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in tqdm(
        enumerate(training_loader),
        total=len(training_loader),
        desc=f"Training Epoch {epoch_index}",
    ):
        # Every data instance is an input + label pair
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def train_model(
    training_loader: DataLoader,
    validation_loader: DataLoader,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    epochs: int,
    experiment_name: str,
    checkpoint_dir: Path = Settings.PROJECT_PATH / "models",
    tensorboard_dir: Path = Settings.PROJECT_PATH / "runs",
    best_vloss: float = float("inf"),
    starting_epoch: int = 0,
    early_stop: EarlyStop = None,
    device: DeviceLikeType = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
):
    if early_stop is not None:
        early_stop.min_validation_loss = best_vloss
    checkpoint_dir_path = checkpoint_dir / experiment_name
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(str(tensorboard_dir / experiment_name))

    for epoch in range(starting_epoch + 1, epochs + 1):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            training_loader, model, optimizer, loss_fn, epoch, device, tb_writer
        )

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in tqdm(
                enumerate(validation_loader),
                total=len(validation_loader),
                desc="Calculating vloss",
                leave=False,
            ):
                vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        tb_writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch,
        )
        tb_writer.flush()

        checkpoint_path = save_checkpoint(locals(), checkpoint_dir_path)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_checkpoint_path = checkpoint_dir_path / "best_checkpoint.pth"
            best_checkpoint_path.unlink(True)
            best_checkpoint_path.symlink_to(checkpoint_path)

        if early_stop is not None:
            if early_stop(avg_vloss):
                print(
                    f"Early stopping! vloss not decreasing for {early_stop.patience} epochs."
                )
                return


def save_checkpoint(state: Dict[str, Any], checkpoint_dir: Path) -> Path:
    model_state = {
        "state_dict": state["model"].state_dict(),
        "optimizer": state["optimizer"].state_dict(),
        "epoch": state["epoch"],
        "train_loss": state["avg_loss"],
        "validation_loss": state["avg_vloss"],
        "experiment_name": state["experiment_name"],
    }
    checkpoint_path = checkpoint_dir / f"checkpoint_{state['epoch']}.pth"
    torch.save(model_state, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path, model: Module, optimizer: Optimizer
) -> Tuple[int, float, str]:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"], checkpoint["train_loss"], checkpoint["experiment_name"]
