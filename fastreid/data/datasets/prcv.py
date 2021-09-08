# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os

from scipy.io import loadmat

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['PRCV', ]


@DATASET_REGISTRY.register()
class PRCV(ImageDataset):
    dataset_name = "prcv"
    dataset_dir = "prcv"

    def __init__(self, root='datasets', **kwargs):
        self.root = root

        self.train_path = os.path.join(self.root, self.dataset_dir, "train/training_images")
        self.label_path = os.path.join(self.root, self.dataset_dir, "train/RAP_reid_data.mat")
        self.test_path = os.path.join(self.root, self.dataset_dir, "prcv_test")

        required_files = [self.train_path, self.label_path, self.test_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path, self.label_path)
        test = self.process_test(self.test_path)
        train.extend(test)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path, label_path):
        img_paths = []

        all_labels = loadmat(label_path)['RAP_reid_data'][0, 0][0]
        for info in all_labels:
            img_path = os.path.join(train_path, info[0].item())
            pid = self.dataset_name + "_" + str(info[1].item())
            camid = self.dataset_name + "_" + info[2].item()[3:]
            img_paths.append([img_path, pid, camid])

        return img_paths

    def process_test(self, dir_path):
        data = []
        for dir_name in os.listdir(dir_path):
            img_lists = glob.glob(os.path.join(dir_path, dir_name, "*.png"))
            for img_path in img_lists:
                pid = self.dataset_name + "_test" + dir_name.split('-')[-1]
                camid = self.dataset_name + "_" + "0"
                data.append([img_path, pid, camid])
        return data
