"""
Contains functions for training and testing a PyTorch model.
"""

from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm
from torchmetrics import AUROC
import math


from common.utils import print_train_time, EarlyStopping, AUCTracker


# Function to print the learning rate
def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]  # Get the lr of the first parameter group


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Trains a PyTorch model for a single epoch.

    Returns:
    A tuple of (train_loss, train_accuracy, train_auc).
    """
    model.train()
    train_loss, train_acc = 0, 0
    auroc = AUROC(pos_label=1).to(device)  # TorchMetrics AUROC
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred_logits = model(X).squeeze()
        loss = loss_fn(y_pred_logits, y.float())
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Predictions
        y_pred_probs = torch.sigmoid(y_pred_logits)  # Convert logits to probabilities
        y_pred_class = torch.round(y_pred_probs)

        train_acc += (y_pred_class == y).sum().item() / len(y)

        # Update AUC metric
        auroc.update(y_pred_probs, y.int())

        if scheduler:
            scheduler.step()
            if i % 5000 == 0:
                print(f"Iteração {i+1}, LR: {optimizer.param_groups[0]['lr']}")

    # Compute metrics
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    train_auc = auroc.compute().item()  # Get AUC score

    return train_loss, train_acc, train_auc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Tests a PyTorch model for a single epoch.

    Returns:
    A tuple of (test_loss, test_accuracy, test_auc).
    """
    model.eval()
    test_loss, test_acc = 0, 0
    auroc = AUROC(pos_label=1).to(device)  # TorchMetrics AUROC

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred_logits = model(X).squeeze()
            loss = loss_fn(test_pred_logits, y.float())
            test_loss += loss.item()

            # Predictions
            test_pred_probs = torch.sigmoid(test_pred_logits)
            test_pred_labels = torch.round(test_pred_probs)

            test_acc += (test_pred_labels == y).sum().item() / len(y)

            # Update AUC metric
            auroc.update(test_pred_probs, y.int())

    # Compute metrics
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    test_auc = auroc.compute().item()  # Get AUC score

    return test_loss, test_acc, test_auc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    val_target: torch.utils.data.DataLoader,
    test_target: torch.utils.data.DataLoader,
    val_target_2: torch.utils.data.DataLoader,
    test_target_2: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    early_stop_patience: int,
    exp,
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """
    # Make sure model on target device
    model.to(device)

    early_stopper = EarlyStopping(patience=early_stop_patience, mode="max")

    # Variables to track best accuracy
    best_val_metric = 0
    auc_tracker = AUCTracker(window_size=5)
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_auc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        val_loss, val_acc, val_auc = test_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )
        auc_tracker.update(val_auc)
        auc_smooth = auc_tracker.moving_average()

        # Save model if validation accuracy improves
        if auc_smooth > best_val_metric:
            best_val_metric = auc_smooth
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_metric": auc_smooth,
                },
                checkpoint_path,
            )
            # print(f"Model saved with validation auc: {best_val_metric:.2f}%")

        # Update results dictionary
        exp.log_metrics(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auc": train_auc,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_acc": val_acc,
            }
        )
        if early_stopper.step(auc_smooth):
            # print(f"We are at epoch: {epoch}")
            break

    # Return the filled results at the end of the epochs
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_val_metric = checkpoint["best_val_metric"]
    start_epoch = checkpoint["epoch"]

    exp.log_model(
        name="model_state_dict",
        file_or_folder=checkpoint_path,
        metadata={"framework": "pytorch"},
    )

    exp.log_other("best_val_metric", best_val_metric)
    exp.log_other("best_epoch", start_epoch)
    print(
        f"Loaded model from epoch {start_epoch} with best validation auc: {best_val_metric:.2f}%"
    )
    test_loss, test_acc, test_auc = test_step(
        model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
    )
    test_loss_target, test_acc_target, test_auc_target = test_step(
        model=model, dataloader=test_target, loss_fn=loss_fn, device=device
    )
    test_loss_target_2, test_acc_target_2, test_auc_target_2 = test_step(
        model=model, dataloader=test_target_2, loss_fn=loss_fn, device=device
    )

    exp.log_metrics(
        {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_auc": test_auc,
            "test_loss_target": test_loss_target,
            "test_acc_target": test_acc_target,
            "test_auc_target": test_auc_target,
            "test_loss_target_2": test_loss_target_2,
            "test_acc_target_2": test_acc_target_2,
            "test_auc_target_2": test_auc_target_2,
        }
    )

    print("DONE")
    # End the current experiment
    exp.end()
