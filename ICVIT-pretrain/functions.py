import numpy as np
import torch
import torch.nn.functional as F

import sys
from sklearn import metrics
from tqdm import tqdm
import math

import dino_utils
import wandb

def train_one_epoch_dino(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, epochs, clip_grad=3.0):
    metric_logger = dino_utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, epochs)
    for it, (images, cls_mask) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.amp.autocast('cuda', enabled=fp16_scaler is not None):
            teacher_output = teacher(images[:2], cls_mask[:2])  # only the 2 global views pass through the teacher
            student_output = student(images, cls_mask)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if clip_grad > 0:
                param_norms = dino_utils.clip_gradients(student, clip_grad)
            dino_utils.cancel_gradients_last_layer(epoch, student, 1)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if clip_grad > 0:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = dino_utils.clip_gradients(student, clip_grad)
            dino_utils.cancel_gradients_last_layer(epoch, student, 1)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def train_classifier(
        train_dataloader, 
        val_dataloader, 
        model, 
        optimizer, 
        lr_schedule, 
        wd_schedule, 
        epochs=100, 
        logger=None, 
        device='cuda', 
        run_name='default',
        val_epochs=1,
        val_only=False):
    logger.info(f'Training start: {run_name}')

    if val_only:
        results = val(val_dataloader, model, device)
        logger.info(f"[Validation || Accuracy: {results['Accuracy']:.4f}")
        logger.info("Validation finish!")
        return None

    for epoch in range(1, epochs+1):
        for i, data in enumerate(train_dataloader):
            it = len(train_dataloader) * (epoch-1) + i  # global training iteration
            for idx, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if idx == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

            image = data['image'].to(device)
            label = data['label'].to(device)#.float()
            channels = data['channels']

            pred = model(image, extra_tokens={'n_channels': channels})
            loss = F.cross_entropy(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                logger.info(f'Epoch: {epoch:03d}, ' +
                    f'iter: {i:05d}/{len(train_dataloader):05d}, ' + 
                    f'lr: {optimizer.param_groups[0]["lr"]:.5f}, ' + 
                    f'wd: {optimizer.param_groups[0]["weight_decay"]:.4f}, ' + 
                    f'loss: {loss:.6f}')
                wandb.log({"train/loss": loss.item()})

        if epoch % val_epochs == 0:
            results = val(val_dataloader, model, device)
            logger.info(f"[Validation || Epoch: {epoch:03d} | accuracy: {results['Accuracy']:.4f}")
            torch.save(model.state_dict(), f'logs/{run_name}/checkpoints/model_{epoch}.pth')

            wandb.log({"epoch": epoch, "test/acc": results['Accuracy']})

    logger.info("Training finish!")


def val(dataloader, model, device='cuda'):
    model.eval()

    labels, preds  = [], []
    for data in tqdm(dataloader):
        X = data['image'].to(device)
        y = data['label'].to(device).float()
        channels = data['channels']

        with torch.no_grad():
            # y_ = model(X).squeeze(1)
            # score = torch.sigmoid(y_)
            y_ = model(X, extra_tokens={'n_channels': channels}).softmax(dim=1)

        labels.extend(y.tolist())
        # pred = (score > 0.5).int()
        pred = torch.argmax(y_, dim=1)
        preds.extend(pred.tolist())

        # if len(preds) > 1000:
        #     break

    # precision = metrics.precision_score(labels, preds)
    # recall = metrics.recall_score(labels, preds)
    # f1 = metrics.f1_score(labels, preds)
    # accuracy = metrics.accuracy_score(labels, preds)
    accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)

    results = {
        # 'Precision': precision,
        # 'Recall': recall,
        # 'F1-score': f1,
        'Accuracy': accuracy
    }

    model.train()
    return results




