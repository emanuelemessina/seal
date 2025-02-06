import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

from load_checkpoint import load_checkpoint


def train(device, model, multiscale_roi_align, dataset, dataloader, batch_size, checkpoint_path, discard_optim):
    '''
    # split params for different learning rates (optional)

    box_regression_params = []
    classification_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bbox_pred" in name:
            box_regression_params.append(param)
        else:
            classification_params.append(param)
    '''

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

    ''' linear model
    lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    iterations_per_epoch = (len(dataset) + batch_size - 1) // batch_size
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=iterations_per_epoch, T_mult=2, eta_min=0.0001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    '''
    # conv model
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = None

    load_checkpoint(checkpoint_path, discard_optim, model, optimizer, scheduler)

    loss_fn_superclass = CrossEntropyLoss(weight=dataset.superclass_weights.to(device))
    loss_fn_class = CrossEntropyLoss(weight=dataset.class_weights.to(device))
    lambda_superclasses = 0.99
    lambda_classes = 0.95

    # Training loop

    model.train()

    date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = open(f'log_{date_time}.txt', 'a')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')

    loss_file = open(f'loss_{date_time}.csv', 'a')

    loss_file.write(
        'epoch,batch,rpn_localization_loss,rpn_classification_loss,frcnn_localization_loss,frcnn_classification_loss')
    loss_file.write(',custom_classification_super_loss,custom_classification_sub_loss')
    loss_file.write('\n')

    log(f'Started training {date_time}')
    log(f'lr: {lr}')

    num_epochs = 10
    for epoch in range(num_epochs):

        log(f'Epoch {epoch} ...')

        for batch_idx, (images, targets) in enumerate(dataloader):

            log(f'Batch {batch_idx + 1}/{len(dataloader)}...')

            rpn_classification_losses = []
            rpn_localization_losses = []
            frcnn_classification_losses = []
            frcnn_localization_losses = []
            custom_classification_superlosses = []
            custom_classification_sublosses = []

            optimizer.zero_grad()

            gt_boxes = []
            gt_sublabels = []
            gt_superlabels = []
            for idx, target in enumerate(targets):
                target['boxes'] = target['boxes'].float().to(device)
                gt_boxes.append(target['boxes'])
                target['superlabels'] = target['superlabels'].long().to(device)
                target['sublabels'] = target['labels'].long().to(device)
                target['labels'] = torch.ones_like(target['labels']).long().to(
                    device)  # dummy labels800 for the binary classifier
                gt_sublabels.append(target['sublabels'])
                gt_superlabels.append(target['superlabels'])

            images = [image.float().to(device) for image in images]

            batch_losses = model(images, targets)

            superloss = subloss = 0

            features = model.roi_heads.box_roi_pool.features
            image_shapes = model.roi_heads.box_roi_pool.image_shapes
            box_features = multiscale_roi_align(features, gt_boxes, image_shapes)
            box_head = model.roi_heads.box_head
            x1 = box_head(box_features)
            custom_classifier = model.roi_heads.box_predictor.custom_forward
            super_logits, sub_logits = custom_classifier(x1)
            gt_sublabels = torch.cat(gt_sublabels, dim=0)
            gt_superlabels = torch.cat(gt_superlabels, dim=0)
            superloss = lambda_superclasses * loss_fn_superclass(super_logits, gt_superlabels)
            subloss = lambda_classes * loss_fn_class(sub_logits, gt_sublabels)

            loss = batch_losses['loss_box_reg']
            loss += batch_losses['loss_rpn_box_reg']
            loss += batch_losses['loss_objectness']
            loss += batch_losses['loss_classifier']

            loss += superloss + subloss

            loss.backward()

            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())

            custom_classification_superlosses.append(superloss.item())
            custom_classification_sublosses.append(subloss.item())

            optimizer.step()

            if batch_idx % 10 == 0:
                rpn_classification_mean = np.mean(rpn_classification_losses)
                rpn_localization_mean = np.mean(rpn_localization_losses)
                frcnn_localization_mean = np.mean(frcnn_localization_losses)

                loss_output = ''
                loss_output += f'{"RPN Localization Loss":<26}: {rpn_localization_mean:.20f}\n'
                loss_output += f'{"RPN Classification Loss":<26}: {rpn_classification_mean:.20f}\n'
                loss_output += f'{"Head Localization Loss":<26}: {frcnn_localization_mean:.20f}\n'
                frcnn_classification_mean = np.mean(frcnn_classification_losses)
                loss_output += f'{"Head Classification Loss":<26}: {frcnn_classification_mean:.20f}\n'

                custom_classification_supermean = np.mean(custom_classification_superlosses)
                custom_classification_submean = np.mean(custom_classification_sublosses)
                loss_output += f'{"Super Classification Loss":<26}: {custom_classification_supermean:.20f}\n'
                loss_output += f'{"Sub Classification Loss":<26}: {custom_classification_submean:.20f}\n'

                log(loss_output)

                loss_file.write(
                    f'{epoch},{batch_idx},{rpn_localization_mean:.20f},{rpn_classification_mean:.20f},{frcnn_localization_mean:.20f},{frcnn_classification_mean:.20f}')

                loss_file.write(f',{custom_classification_supermean:.20f},{custom_classification_submean:.20f}')
                loss_file.write('\n')

            if batch_idx != 0 and batch_idx % (len(dataset) // (batch_size * 2) - 1) == 0:
                # save state
                checkpoint_name = f'checkpoint_e{epoch}_b{batch_idx}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.pth'
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()},
                           checkpoint_name)
                log(f"Saved checkpoint {checkpoint_name}")

            #scheduler.step(epoch + batch_idx / iterations_per_epoch)
        if scheduler:
            scheduler.step()
            log(f"lr: {scheduler.get_last_lr()}")

    log("Training done.")
    log_file.close()
    loss_file.close()
