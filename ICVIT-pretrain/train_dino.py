import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

#from data.base_dataset import DINODataset
from data.so2sat import So2Sat
from data.sat_base import SatBase
from archs import build_model
from archs.hf_vit import DINOHead
from functions import train_one_epoch_dino
import dino_utils
import datetime 
import argparse
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    logger = create_logger(args.run_name)

    dino_utils.init_distributed_mode(args)
    dino_utils.fix_random_seeds(args.seed)
    # print("git:\n  {}\n".format(dino_utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    transforms = dino_utils.DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )

    dataset = So2Sat(args.path_root, transform=transforms)
    #dataset = SatBase(args.path_root, transform=transforms)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=args.batch_size, 
        num_workers=8, drop_last=True)

    student = build_model(
        args.network, image_size=args.image_size,
        channel=args.channel, n_classes=0, drop_path_rate=0.1, pretrained=args.pretrained)

    teacher = build_model(
        args.network, image_size=args.image_size, 
        channel=args.channel, n_classes=0, pretrained=args.pretrained)

    embed_dim = student.embed_dim
    logger.info(f'Embed_dim: {embed_dim}')

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = dino_utils.MultiCropWrapper(
        student, DINOHead(embed_dim, 65536, use_bn=False, norm_last_layer=True)
    )
    teacher = dino_utils.MultiCropWrapper(
        teacher, DINOHead(embed_dim, 65536, use_bn=False)
    )
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if dino_utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.gpu], find_unused_parameters=args.network=='dinov2')
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(
        student, device_ids=[args.gpu], find_unused_parameters=args.network=='dinov2')
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info(f"Student and Teacher are built: they are both {args.network} network.")

    # ============ preparing loss ... ============
    dino_loss = dino_utils.DINOLoss(
        out_dim=65536, 
        ncrops=args.local_crops_number + 2, 
        warmup_teacher_temp=0.04, 
        teacher_temp=0.04, 
        warmup_teacher_temp_epochs=0, 
        nepochs=args.epochs,
    ).cuda()

    params_groups = dino_utils.get_params_groups(student)
    optimizer = optim.AdamW(params_groups)

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = dino_utils.cosine_scheduler(
        base_value=args.lr*(args.batch_size*dino_utils.get_world_size())/256.,  # linear scaling rule
        final_value=1e-6,
        epochs=args.epochs, 
        niter_per_ep=len(data_loader),
        warmup_epochs=10,
    )
    wd_schedule = dino_utils.cosine_scheduler(
        base_value=0.04,
        final_value=0.4,
        epochs=args.epochs, 
        niter_per_ep=len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = dino_utils.cosine_scheduler(0.996, 1, args.epochs, len(data_loader))
    logger.info(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    dino_utils.restart_from_checkpoint(
        os.path.join(f"logs/{args.run_name}", "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    logger.info("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch_dino(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args.epochs)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        dino_utils.save_on_master(save_dict, os.path.join(f"logs/{args.run_name}", 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            dino_utils.save_on_master(save_dict, os.path.join(f"logs/{args.run_name}/checkpoints", f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if dino_utils.is_main_process():
            logger.info(log_stats)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path-root', default='~/dataset/wsi/so2sat')
    parser.add_argument('--train-csv', default='~/dataset/wsi/alldata.csv')
    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.), 
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.1, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--local-rank', type=int, default=0)

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=201)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--saveckp-freq', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--channel', type=int, default=1)
    parser.add_argument('--network', default='icvit', choices=['vit', 'dinov1', 'icvit'], help='network name')
    parser.add_argument('--pretrained', action='store_true', help='Whether to use pretrained model')
    parser.add_argument('--use-fp16', action='store_true', help='Whether to use fp16 in training')
    parser.add_argument('--run-name', type=str, default='test')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")

    args = parser.parse_args()

    main(args)
