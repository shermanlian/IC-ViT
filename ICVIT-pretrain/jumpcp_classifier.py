import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.jumpcp import JUMPCPDataset
from archs import build_model
from functions import train_classifier
import dino_utils

import argparse
import logging
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def wandb_init(args):
    name = f"{args.network}_{args.run_name}"
    wandb.init(
        reinit=True,
        project="jumpcp",
        name=name,
        group=f"{args.network}",
        job_type=f"classifier",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "run_name": args.run_name,
            "pretrained":args.pretrained,
        }
    )

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    os.makedirs(f'logs/{logging_dir}', exist_ok=True)
    os.makedirs(f'logs/{logging_dir}/checkpoints', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def main(args):
    if not args.val_only:
        wandb_init(args)
    logger = create_logger(args.run_name)
    torch.manual_seed(args.seed)

    train_dataset = JUMPCPDataset(args.path_root, args.path_pq, split='train')
    val_dataset = JUMPCPDataset(args.path_root, args.path_pq, split='test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    model = build_model(
        args.network, 
        image_size=args.image_size, 
        channel=args.channel, 
        n_classes=args.n_classes, 
        pretrained=args.pretrained).to(device)

    params_groups = dino_utils.get_params_groups(model)
    optimizer = optim.AdamW(params_groups)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # ============ init schedulers ... ============
    lr_schedule = dino_utils.cosine_scheduler(
        base_value=args.lr,  # linear scaling rule
        final_value=1e-6,
        epochs=args.epochs, 
        niter_per_ep=len(train_dataloader),
        warmup_epochs=5,
    )
    wd_schedule = dino_utils.cosine_scheduler(
        base_value=args.wd,
        final_value=0.4,
        epochs=args.epochs, 
        niter_per_ep=len(train_dataloader),
    )

    logger.info(f"Initialized! {args}")
    train_classifier(train_dataloader, val_dataloader, 
        model, optimizer, lr_schedule, wd_schedule, 
        epochs=args.epochs, logger=logger, val_only=args.val_only)

    if not args.val_only:
        wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path-root', default='~/datasets/wsi/jumpcp')
    parser.add_argument('--path-pq', default='~/datasets/wsi/jumpcp/BR00116991.pq')

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=0.04)
    parser.add_argument('--network', default='icvit', help='network name')
    parser.add_argument('--channel', type=int, default=10)
    parser.add_argument('--n_classes', type=int, default=161)
    parser.add_argument('--pretrained', action='store_true', help='Whether to use pretrained model')
    parser.add_argument('--val-only', action='store_true', help='Only perform model validation')
    parser.add_argument('--run-name', type=str, default='default')
    args = parser.parse_args()

    main(args)
