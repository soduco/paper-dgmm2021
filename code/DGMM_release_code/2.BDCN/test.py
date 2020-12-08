import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import time
import os
import bdcn
from datasets.dataset import Data
import argparse
import cfg
from PIL import Image

def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))


def test(model, args):
    test_root = cfg.config_test[args.dataset]['data_root']
    test_lst = cfg.config_test[args.dataset]['data_lst']
    test_name_lst = os.path.join(test_root, 'test.lst')
    mean_bgr = np.array(cfg.config_test[args.dataset]['mean_bgr'])

    test_img = Data(test_root, test_lst, mean_bgr=mean_bgr)
    testloader = torch.utils.data.DataLoader(test_img, batch_size=1, shuffle=False, num_workers=1)
    nm = np.loadtxt(test_name_lst, dtype=str)

    save_dir = os.path.join(args.res_dir, args.model.split('/')[-1].split('.')[0] + '_fuse')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.cuda:
        model.cuda()

    model.eval()
    start_time = time.time()
    all_t = 0
    for i, (data, _) in enumerate(testloader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        tm = time.time()

        with torch.no_grad():
            out = model(data)

        fuse = torch.sigmoid(out[-1]).cpu().numpy()[0, 0, :, :]

        if not os.path.exists(os.path.join(save_dir, 'fuse')):
            os.mkdir(os.path.join(save_dir, 'fuse'))

        fuse = fuse * 255
        fuse = Image.fromarray(fuse).convert('RGB')
        fuse.save(os.path.join(save_dir, 'fuse', '{}.png'.format(nm[i][0].split('/')[2].split('.')[0])))
        all_t += time.time() - tm
    print('Save prediction into folder {}'.format(str(os.path.join(save_dir, 'fuse'))))
    print('Overall Time use: ', time.time() - start_time)

def main():
    args = parse_args()

    # Choose the GPUs
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = bdcn.BDCN()
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    model.load_state_dict(torch.load('%s' % (args.model), map_location=map_location))
    test(model, args)

def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
        default='HistoricalMap2020', help='The dataset to train')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default=None,
        help='the model to test')
    parser.add_argument('--res-dir', type=str, default='results',
        help='the dir to store result')
    return parser.parse_args()

if __name__ == '__main__':
    main()
