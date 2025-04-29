import os
from typing import Dict, List

import torch
from tqdm.auto import tqdm
from loguru import logger

import torch.nn.functional as F
from src.pytorch_adapt.utils.common_functions import batch_to_device

from common.utils import EarlyStopping, ModelCheckpoint, create_directory, AUCTracker


def train_step(
    model: torch.nn.Module,
    dataloaders: torch.utils.data.DataLoader,
    train_name: str = None,
    hook: torch.nn.Module = None,
    misc: torch.nn.Module = None,
    device: torch.device = None,
):
    logger.info(f"Train - {train_name}")
    model.train()
    for data in tqdm(dataloaders[train_name]):
        data = batch_to_device(data, device)
        _, loss = hook({**model, **misc, **data})
    return loss


def gen_data_score(
    model: torch.nn.Module,
    dataloaders: torch.utils.data.DataLoader,
    data_type: str = "target_val_with_labels",
    device: torch.device = None,
    exp=None,
):
    logger.info(f"Eval - {data_type}")
    model.eval()
    G, C = model["G"], model["C"]
    labels, logits, preds = [], [], []
    data_side = data_type.split("_")[0]
    with torch.no_grad():
        for data in tqdm(dataloaders[data_type]):
            data = batch_to_device(data, device)
            logit = C(G(data[f"{data_side}_imgs"]))
            if isinstance(logit, list):
                logit = logit[0]
            pred = F.softmax(logit, dim=-1)
            logits.append(logit)
            preds.append(pred)
            if f"{data_side}_labels" in data:
                label = data[f"{data_side}_labels"]
                labels.append(label)
        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        if labels:
            labels = torch.cat(labels, dim=0)
            data_score = {"logits": logits, "preds": preds, "labels": labels}
        else:
            data_score = {"logits": logits, "preds": preds}
    return data_score


def test_step(
    model: torch.nn.Module,
    dataloaders: torch.utils.data.DataLoader,
    data_split: str = "val",
    validator: torch.nn.Module = None,
    device: torch.device = None,
    exp=None,
):
    src_data_score = gen_data_score(
        model, dataloaders, "src_" + data_split, device, exp
    )
    target_data_score = gen_data_score(
        model, dataloaders, "target_" + data_split + "_with_labels", device, exp
    )
    if validator["Name"] in [
        "InfoMax",
        "Entropy",
    ]:
        dict_scores = {
            key: data_score
            for key, data_score in zip(
                validator["Params"], [src_data_score, target_data_score]
            )
        }
        score = validator["Class"](**dict_scores)
        scores = [score, score]
    elif validator["Name"] in [
        "SND",
    ]:
        dict_scores = {
            key: data_score
            for key, data_score in zip(
                validator["Params"], [target_data_score]
            )
        }
        score = validator["Class"](**dict_scores)
        scores = [score, score]
    elif validator["Name"] in ["Accuracy", "AUC"]:
        scores = []
        for data_score in [src_data_score, target_data_score]:
            dict_scores = {
                key: data_score
                for key, data_score in zip(validator["Params"], [data_score])
            }
            scores.append(validator["Class"](**dict_scores))
    return scores


def train(
    model: torch.nn.Module,
    model_name: str,
    dataloaders: torch.utils.data.DataLoader,
    train_name: str = None,
    optimizer=None,
    schedulers=None,
    hook=None,
    misc=None,
    validator=None,
    exp=None,
    args=None,
) -> Dict[str, List]:
    # Init early_stopper
    if args.early_stop_activate:
        early_stopper = EarlyStopping(
            patience=args.early_stop_patience, mode=args.early_stop_mode
        )

    # Init model_checkpoint
    create_directory(type_path="checkpoints", args=args)
    model_checkpoint = ModelCheckpoint(
        name=f"{model_name}.pth",
        mode=args.checkpoint_mode,
        path_save=os.path.join("checkpoints"),
    )
    auc_tracker = AUCTracker()
    # Start training - Loop through training and validate steps for a number of epochs
    for epoch in range(args.n_epoch):
        logger.info(f"Epoch {epoch+1}/{args.n_epoch}".center(70, "+"))

        train_loss = train_step(
            model=model,
            dataloaders=dataloaders,
            train_name=train_name,
            hook=hook,
            misc=misc,
            device=args.device,
        )
        logger.success(f"Train Source Loss: {train_loss}")

        train_source, train_target = test_step(
            model=model,
            dataloaders=dataloaders,
            data_split="train",
            validator=validator,
            device=args.device,
            exp=exp,
        )

        val_source, val_target = test_step(
            model=model,
            dataloaders=dataloaders,
            data_split="val",
            validator=validator,
            device=args.device,
            exp=exp,
        )
        if args.validator == "AUC":
            auc_smt = auc_tracker.update(val_target)
        logger.success(f"{validator['Name']} from Train Source: {train_source}")
        logger.success(f"{validator['Name']} from Train Target: {train_target}")
        logger.success(f"{validator['Name']} from Val Source: {val_source}")
        logger.success(f"{validator['Name']} from Val Target: {val_target}")

        # Model checkpoint call
        logger.info("Model Checkpoint")
        model_checkpoint(
            model,
            auc_smt if args.validator == "AUC" else val_target,
            epoch=epoch,
            optimizer=optimizer,
            misc=misc,
            args=args,
        )

        # Log to Comet
        if args.validator == "AUC":
            exp.log_metrics(
                {
                    "epoch": epoch,
                    "train_loss": train_loss["total_loss"]["total"],
                    "train_source": train_source,
                    "train_target": train_target,
                    "val_source": val_source,
                    "val_target": val_target,
                    "best_score": model_checkpoint.get_best_score(),
                    "best_epoch": model_checkpoint.get_best_epoch(),
                    "val_source_auc_smt": auc_smt,
                }
            )
        else:
            exp.log_metrics(
                {
                    "epoch": epoch,
                    "train_loss": train_loss["total_loss"]["total"],
                    "train_source": train_source,
                    "train_target": train_target,
                    "val_source": val_source,
                    "best_score": model_checkpoint.get_best_score(),
                    "best_epoch": model_checkpoint.get_best_epoch(),
                    "val_target": val_target,
                }
            )

        # Early stopping call
        if args.early_stop_activate:
            logger.info("Early Stopping")
            if early_stopper.step(val_source):
                logger.warning(f"We are at epoch: {epoch}")
                break

        # schedulers.step("per_epoch")
