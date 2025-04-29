import os
import json
import copy
from argparse import Namespace
import argparse

import comet_ml
import torch
import torch.nn as nn
from loguru import logger
from src.pytorch_adapt.containers import Models, Optimizers, LRSchedulers
from src.pytorch_adapt.hooks import (
    ATDOCHook,
    BNMHook,
    MCCHook,
    ClassifierHook,
    FinetunerHook,
)
from src.pytorch_adapt.layers import (
    RandomizedDotProduct,
    MCCLoss,
    MultipleModels,
)
from src.pytorch_adapt.models import Discriminator
from src.pytorch_adapt.validators import (
    AccuracyValidator,
    AUCValidator,
    IMCombinedValidator,
    SNDValidator,
    EntropyCombinedValidator,
)
from src.pytorch_adapt.weighters import MeanWeighter
from src.pytorch_adapt.utils import common_functions as c_f

from common.utils import set_random_seed
from common.networks import ARCHITECTURES, Classifier
from common.data_setup import create_dataloaders_mri_2d
from common.engine import train


BASE_DIR = os.path.dirname(os.path.abspath("__file__"))


def load_data(args):
    """
    src_domain, tgt_domain data to load
    """
    dataloaders, target_dataset_size, train_name = create_dataloaders_mri_2d(
        source=args.src_domain,
        target=args.target_domain,
        algorithm=args.algorithm,
        validator=args.validator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed_dataloaders,
    )
    return dataloaders, target_dataset_size, train_name


def get_model(args):
    # Get the G
    G = ARCHITECTURES[args.G_arch]["model"](
        weights=ARCHITECTURES[args.G_arch]["weights"]
    )
    if args.G_arch in ["vgg16", "densenet161", "densenet201"]:
        G.classifier = nn.Identity()
    else:
        G.fc = nn.Identity()
    feature_dim = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "vgg16": 25088,
    }[args.G_arch]
    if args.init_source_only:
        logger.info("Init G with Source Only")
        G_state_dict = torch.load(
            os.path.join(
                "best_model",
                f"{args.init_model_name}.pth",
            )
        )["models"]["G"]
        G.load_state_dict(G_state_dict, strict=True)
    G = G.to(args.device)

    hidden_size = {
        "resnet18": 256,
        "resnet34": 256,
        "resnet50": 512,
        "resnet101": 2048,
    }[args.G_arch]

    # Get the C
    C = Classifier(
        in_size=feature_dim,
        hidden_size=hidden_size,
        dropout=args.dropout,
        num_classes=args.num_classes,
    )
    if args.init_source_only:
        logger.info("Init C with Source Only")
        C_state_dict = torch.load(
            os.path.join(
                "best_model",
                f"{args.init_model_name}.pth",
            )
        )["models"]["C"]
        C.load_state_dict(C_state_dict, strict=True)
    if args.algorithm == "mcd":
        C = MultipleModels(C, c_f.reinit(copy.deepcopy(C)))
    C = C.to(args.device)

    # Get the D
    D = Discriminator(in_size=feature_dim)
    D = D.to(args.device)

    feature_combiner = RandomizedDotProduct(
        in_dims=[feature_dim, args.num_classes], out_dim=feature_dim
    )
    misc = {"feature_combiner": feature_combiner}

    return Models({"G": G, "C": C, "D": D}), misc, feature_dim


def get_optimizer(model, args):
    optimizers = Optimizers(
        (
            torch.optim.SGD,
            {"lr": args.lr, "weight_decay": args.weight_decay, "momentum": 0.9},
        ),
        multipliers={"G": 1.0, "C": 1.0, "D": 1.0},
    )
    schedulers = LRSchedulers(
        (torch.optim.lr_scheduler.StepLR, {"step_size": 1, "gamma": 0.1})
    )
    optimizers.create_with(model)
    schedulers.create_with(optimizers)
    optimizers = list(optimizers.values())
    return optimizers, schedulers


def get_hook(optimizers, args):
    HOOKS = {
        "source-only": ClassifierHook(
            loss_fn=torch.nn.CrossEntropyLoss(weight=args.weight),
            opts=[optimizers[0], optimizers[1]],
        ),
        "atdoc": ClassifierHook(
            loss_fn=torch.nn.CrossEntropyLoss(weight=args.weight),
            opts=[optimizers[0], optimizers[1]],
            post=[
                ATDOCHook(
                    dataset_size=args.target_dataset_size,
                    feature_dim=args.feature_dim,
                    num_classes=2,
                    k=args.k_atdoc,
                )
            ],
            weighter=MeanWeighter(
                weights={
                    "pseudo_label_loss": args.lambda_atdoc,
                    "c_loss": args.lambda_L,
                }
            ),
        ),
        "bnm": ClassifierHook(
            loss_fn=torch.nn.CrossEntropyLoss(weight=args.weight),
            opts=[optimizers[0], optimizers[1]],
            post=[BNMHook()],
            weighter=MeanWeighter(
                weights={
                    "bnm_loss": args.lambda_bnm,
                    "c_loss": args.lambda_L,
                }
            ),
        ),
        "mcc": ClassifierHook(
            loss_fn=torch.nn.CrossEntropyLoss(weight=args.weight),
            opts=[optimizers[0], optimizers[1]],
            post=[MCCHook(loss_fn=MCCLoss(T=args.T_mcc))],
            weighter=MeanWeighter(
                weights={
                    "mcc_loss": args.lambda_mcc,
                    "c_loss": args.lambda_L,
                }
            ),
        ),
    }
    return HOOKS[args.algorithm]


def get_validators(args):
    VALIDATORS = {
        "Accuracy": {
            "Name": "Accuracy",
            "Class": AccuracyValidator(),
            "Params": ["src_val"],
        },
        "AUC": {
            "Name": "AUC",
            "Class": AUCValidator(),
            "Params": ["src_val"],
        },
        "InfoMax": {
            "Name": "InfoMax",
            "Class": IMCombinedValidator(),
            "Params": ["src_train", "target_train"],
        },
        "SND": {
            "Name": "SND",
            "Class": SNDValidator(),
            "Params": ["target_train"],
        },
        "Entropy": {
            "Name": "Entropy",
            "Class": EntropyCombinedValidator(),
            "Params": ["src_train", "target_train"],
        },
    }
    return VALIDATORS[args.validator]


def main(exp):
    args = Namespace()
    args.n_epoch = exp.get_parameter("n_epoch")
    # Early Stop Configs
    args.early_stop_activate = False
    args.early_stop_patience = 5
    args.early_stop_mode = "max"

    # Checkpoints Configs
    args.model_name = exp.get_name()
    args.checkpoint_mode = "max"

    # Define the device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set Random Seed
    args.seed_dataloaders = exp.get_parameter("seed")
    set_random_seed(exp.get_parameter("seed"))

    # Load Data
    args.src_domain = exp.get_parameter("source")
    args.target_domain = exp.get_parameter("target")
    args.algorithm = exp.get_parameter("algorithm")
    args.validator = exp.get_parameter("validator")
    args.batch_size = exp.get_parameter("batch_size")
    args.num_workers = 8

    dataloaders, target_dataset_size, train_name = load_data(args)
    args.target_dataset_size = target_dataset_size
    args.train_name = train_name

    # Get the model
    args.init_model_name = exp.get_parameter("init_model_name")
    args.G_arch = exp.get_parameter("G_arch")
    args.dropout = exp.get_parameter("dropout")
    args.init_source_only = True
    args.num_classes = 2

    model, misc, feature_dim = get_model(args)
    args.feature_dim = feature_dim

    # Get the Optimizer
    args.lr = exp.get_parameter("lr")
    args.weight_decay = exp.get_parameter("weight_decay")
    optimizer, schedulers = get_optimizer(model, args)

    # Get the Hook
    args.k_atdoc = exp.get_parameter("k_atdoc")
    args.lambda_atdoc = exp.get_parameter("lambda_atdoc")
    args.lambda_L = exp.get_parameter("lambda_L")
    args.lambda_bnm = exp.get_parameter("lambda_bnm")
    args.T_mcc = exp.get_parameter("T_mcc")
    args.lambda_mcc = exp.get_parameter("lambda_mcc")

    ## Get Weights for the classes
    if "GE" in args.src_domain:
        total = 5472 + 8640
        w0 = total / 5472
        w1 = total / 8640
        weight = torch.tensor([w0, w1]).to(args.device)
        exp.log_parameter("weight", weight)
    elif "Siemens" in args.src_domain:
        total = 4512 + 7808
        w0 = total / 4512
        w1 = total / 7808
        weight = torch.tensor([w0, w1]).to(args.device)
        exp.log_parameter("weight", weight)
    elif "Philips" in args.src_domain:
        total = 1792 + 2688
        w0 = total / 1792
        w1 = total / 2688
        weight = torch.tensor([w0, w1]).to(args.device)
        exp.log_parameter("weight", weight)

    args.weight = weight
    hook = get_hook(optimizer, args)

    # Get the validador
    validator = get_validators(args)

    # Print the args complete
    logger.info(args)
    # Train Model
    train(
        model=model,
        model_name=args.model_name,
        dataloaders=dataloaders,
        train_name=train_name,
        optimizer=optimizer,
        schedulers=schedulers,
        hook=hook,
        misc=misc,
        validator=validator,
        exp=exp,
        args=args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Name Parser")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment")
    init_code = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
    NAME = init_code.exp_name
    with open(os.path.join(BASE_DIR, "runs", f"{NAME}.json"), "r") as file:
        configs = json.load(file)
    comet_ml.login(project_name=f"ADNI1-{NAME}")
    opt = comet_ml.Optimizer(config=configs, verbose=0)
    for exp in opt.get_experiments():
        main(exp)
        exp.end()
        logger.info("*" * 25)
        logger.info(f"End training")
        logger.info("*" * 25)
