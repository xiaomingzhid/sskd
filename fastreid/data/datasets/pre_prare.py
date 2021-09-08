import os
from os import path

from scipy.io import loadmat
from torchvision.datasets.utils import download_url

sop_dir = path.join('datasets', 'Stanford_Online_Products')
train_file = 'train.txt'
test_file = 'test.txt'

def generate_sop_train_test(sop_dir, train_file, test_file):
    original_train_file = path.join(sop_dir, 'Ebay_train.txt')
    original_test_file = path.join(sop_dir, 'Ebay_test.txt')
    train_file = path.join(sop_dir, train_file)
    test_file = path.join(sop_dir, test_file)

    with open(original_train_file) as f_images:
        train_lines = f_images.read().splitlines()[1:]
    with open(original_test_file) as f_images:
        test_lines = f_images.read().splitlines()[1:]

    train = [','.join((l.split()[-1], str(int(l.split()[1]) - 1))) for l in train_lines]
    test = [','.join((l.split()[-1], str(int(l.split()[1]) - 1))) for l in test_lines]

    with open(train_file, 'w') as f:
        f.write('\n'.join(train))
    with open(test_file, 'w') as f:
        f.write('\n'.join(test))

generate_sop_train_test(sop_dir, train_file, test_file)