import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import time
import os
import bdcn
from datasets.dataset import Data
import cfg
import log
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def adjust_learning_rate(optimizer, steps, step_size, gamma=0.1, logger=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma
        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))


def cross_entropy_loss2d(inputs, targets, cuda=True, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
    inputs = torch.sigmoid(inputs)
    loss = nn.BCELoss(weights, reduction='sum')(inputs, targets)
    return loss


def train(model, args):
    logger = args.logger

    # Training dataloader
    data_root = cfg.config[args.dataset]['data_root']
    data_lst = cfg.config[args.dataset]['data_lst']
    mean_bgr = np.array(cfg.config[args.dataset]['mean_bgr'])
    train_img = Data(data_root, data_lst, mean_bgr=mean_bgr)
    trainloader = torch.utils.data.DataLoader(train_img, batch_size=args.batch_size, shuffle=True, num_workers=20)
    n_train = len(trainloader)
    
    # adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    # Validation dataloader
    val_root = cfg.config_val[args.dataset]['data_root']
    val_lst = cfg.config_val[args.dataset]['data_lst']
    mean_bgr = np.array(cfg.config_val[args.dataset]['mean_bgr'])
    val_img = Data(val_root, val_lst, mean_bgr=mean_bgr)
    valloader = torch.utils.data.DataLoader(val_img, batch_size=args.batch_size, shuffle=False, num_workers=20)

    start_time = time.time()
    if args.cuda:
        model.cuda()
    if args.resume:
        state = torch.load(args.resume)
        optimizer.load_state_dict(state['solver'])
        model.load_state_dict(state['param'])

    batch_size = args.batch_size

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    epochs = args.epochs
    batch_index = 0
    for epoch in range(epochs):
        # Train mode
        model.train()
        mean_loss = []
        val_mean_loss = []
        batch_loss = 0
        pos = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for images, labels in trainloader:
                # Set the gradient in the model into 0
                optimizer.zero_grad()

                if batch_index != batch_size:
                    # If batchsize not equal to batch index , calculate the current loss
                    if args.cuda:
                        images, labels = images.cuda(), labels.cuda()
                    images, labels = Variable(images), Variable(labels)
                    out = model(images)
                    loss = 0
                    # Loss function for 10 different intermediate output
                    for k in range(10):
                        loss += args.side_weight*cross_entropy_loss2d(out[k], labels, args.cuda, args.balance) / (10 * batch_size)

                    # loss function for fuse output
                    loss += args.fuse_weight*cross_entropy_loss2d(out[-1], labels, args.cuda, args.balance) / (10 * batch_size)

                    # Back calculating loss
                    loss.backward()

                    batch_index += 1

                    # Update batch loss
                    batch_loss += loss

                if batch_index == batch_size:
                    # If batchsize equal to batch index , backward the loss and update the loss function
                    # Set batch index to 0
                    batch_index = 0

                    # update parameter, gradient descent, back propagation
                    optimizer.step()

                    # Update the pbar
                    pbar.update(images.shape[0])

                    # Append loss into mean_loss list
                    # Smooth the loss
                    if len(mean_loss) < args.average_loss:
                        mean_loss.append(batch_loss)
                    else:
                        mean_loss[pos] = batch_loss
                        pos = (pos + 1) % args.average_loss

                    batch_loss = 0

                # Add loss (batch) value to tqdm
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Adjust learning rate
        if (epoch+1) % args.step_size == 0:
            adjust_learning_rate(optimizer, epoch+1, args.step_size, args.gamma)

        # Save BDCN weights
        if (epoch+1) % args.snapshots == 0:
            torch.save(model.state_dict(), '%s/bdcn_%d.pth' % (args.param_dir, epoch+1))
            state = {'step': epoch+1, 'param':model.state_dict(),'solver':optimizer.state_dict()}
            torch.save(state, '%s/bdcn_%d.pth.tar' % (args.param_dir, epoch+1))

        tm = time.time() - start_time

        # Evaluate mode
        model.eval()

        for val_images, val_labels in valloader:
            if args.cuda:
                val_images, val_labels = val_images.cuda(), val_labels.cuda()
            val_images, val_labels = Variable(val_images), Variable(val_labels)

            with torch.no_grad():
                out = model(val_images)

            val_loss = 0
            for k in range(10):
                val_loss += args.side_weight * cross_entropy_loss2d(out[k], val_labels, args.cuda, args.balance) / 10
            val_loss += args.fuse_weight * cross_entropy_loss2d(out[-1], val_labels, args.cuda, args.balance) / 10

            val_mean_loss.append(val_loss)

        # Add scalar to tensorboard Loss/Train
        writer.add_scalars('Loss/train/val', {
                            'Train loss': torch.mean(torch.stack(mean_loss)),
                            'Validation loss': torch.mean(torch.stack(val_mean_loss))
        }, epoch)

        logger.info('lr: %e, loss: %f, validation loss: %f, time using: %f' %
                 (optimizer.param_groups[0]['lr'],
                 torch.mean(torch.stack(mean_loss)), torch.mean(torch.stack(val_mean_loss)), tm))

        start_time = time.time()


def main():
    args = parse_args()

    # Choose the GPUs
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger = log.get_logger(args.log)
    args.logger = logger
    logger.info('*'*80)
    logger.info('the args are the below')
    logger.info('*'*80)

    for x in args.__dict__:
        logger.info(x+','+str(args.__dict__[x]))
    logger.info('*'*80)

    if not os.path.exists(args.param_dir):
        os.mkdir(args.param_dir)

    torch.manual_seed(int(time.time()))
    model = bdcn.BDCN(pretrain=args.pretrain, logger=logger)

    if args.complete_pretrain:
        model.load_state_dict(torch.load(args.complete_pretrain))

    train(model, args)

def parse_args():
    parser = argparse.ArgumentParser(description='Train BDCN for different args')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
        default='HistoricalMap2020', help='The dataset to train')
    parser.add_argument('--param-dir', type=str, default='params',
        help='the directory to store the params')
    parser.add_argument('--lr', dest='base_lr', type=float, default=5e-5,
        help='the base learning rate of model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0.002,
        help='the weight_decay of net')
    parser.add_argument('-r', '--resume', type=str, default=None,
        help='whether resume from some, default is None')
    parser.add_argument('-p', '--pretrain', type=str, default=None,
        help='init net from pretrained model default is None')
    parser.add_argument('--epochs', type=int, default=1000,
        help='Epoch to train network, default is 1000')
    parser.add_argument('--average_loss', type=int, default=50,
        help='smoothed loss, default is 50')
    parser.add_argument('-s', '--snapshots', type=int, default=10,
        help='how many iters to store the params, default is 1000')
    parser.add_argument('--step-size', type=int, default=500,
        help='the number of iters to decrease the learning rate, default is 10000')
    parser.add_argument('-b', '--balance', type=float, default=1.1,
        help='the parameter to balance the neg and pos, default is 1.1')
    parser.add_argument('-l', '--log', type=str, default='log.txt',
        help='the file to store log, default is log.txt')
    parser.add_argument('--batch-size', type=int, default=1,
        help='batch size of one iteration, default 1')
    parser.add_argument('--complete-pretrain', type=str, default=None,
        help='finetune on the complete_pretrain, default None')
    parser.add_argument('--side-weight', type=float, default=0.5,
        help='the loss weight of sideout, default 0.5')
    parser.add_argument('--fuse-weight', type=float, default=1.1,
        help='the loss weight of fuse, default 1.1')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='the decay of learning rate, default 0.1')
    return parser.parse_args()

if __name__ == '__main__':
    main()
