import pickle
import numpy as np
import random
import torch
import os
import sys


TMP_DIR = {
    'esc': './tmp/esc',
    'cima': './tmp/cima',
    'cb': './tmp/cb',
}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def load_dataset(data_name):
    dataset = {'train':[], 'test':[], 'valid':[]}
    for key in dataset:
        with open("../data/%s-%s.txt"%(data_name, key),'r') as infile:
            for line in infile:
                dataset[key].append(eval(line.strip('\n')))
    return dataset


def set_cuda(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    devices_id = [int(device_id) for device_id in args.gpu.split()]
    device = (
        torch.device("cuda:{}".format(str(devices_id[0])))
        if use_cuda
        else torch.device("cpu")
    )
    return device, devices_id


def save_rl_mtric(dataset, filename, epoch, SR, mode='train'):
    PATH = TMP_DIR[dataset] + '/eval_result/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/eval_result/'):
        os.makedirs(TMP_DIR[dataset] + '/eval_result/')
    if mode == 'train':
        with open(PATH, 'a') as f:
            f.write('===========Train===============\n')
            f.write('Starting {} user epochs\n'.format(epoch))
            f.write('training SR: {}\n'.format(SR[0]))
            f.write('training Avg@T: {}\n'.format(SR[1]))
            f.write('training Rewards: {}\n'.format(SR[2]))
            f.write('================================\n')
            # f.write('1000 loss: {}\n'.format(loss_1000))
    elif mode == 'test':
        with open(PATH, 'a') as f:
            f.write('===========Test===============\n')
            f.write('Testing {} user tuples\n'.format(epoch))
            f.write('Testing SR: {}\n'.format(SR[0]))
            f.write('Testing Avg@T: {}\n'.format(SR[1]))
            f.write('Testing Rewards: {}\n'.format(SR[2]))
            f.write('================================\n')