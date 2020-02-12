# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from conf import settings
from utils_ai import build_network, get_training_dataloader, get_test_dataloader, WarmUpLR

def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))

        #update training loss for each iteration
        # writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        # writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)
    ))
    print()

    #add informations to tensorboard
    # writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
    # writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_set', type=str, required=True, help='dataset folder')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=48, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args_dict = vars(parser.parse_args())
    net_type = args_dict['net']
    use_gpu = args_dict['gpu']
    train_path=os.path.join(args_dict['data_set'], "train")
    val_path=os.path.join(args_dict['data_set'], "val")
    test_path=os.path.join(args_dict['data_set'], "test")

    net = build_network(archi = net_type, use_gpu=use_gpu, num_classes=num_classes) 
    #data preprocessing:
    training_loader = get_training_dataloader(
        settings.TRAIN_MEAN,
        settings.TRAIN_STD,
        train_path,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    test_loader, idx_to_class = get_test_dataloader(
        settings.TRAIN_MEAN,
        settings.TRAIN_STD,
        test_path,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = './checkpoint/results'
    
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    # writer = SummaryWriter(log_dir=os.path.join(
    #         settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01 
        if best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            logger.info("Saving at epoch: ", epoch, " with accuracy: ", acc)