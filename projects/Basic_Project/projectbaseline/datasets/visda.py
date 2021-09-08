# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os

from fastreid.data.datasets.bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VisDAreid(ImageDataset):
    dataset_dir = "visda"
    dataset_name = "visdareid"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, "pseudo_train")
        self.validation_dir = os.path.join(self.dataset_dir, "target_validation")
        self.test_dir = os.path.join(self.dataset_dir, "target_test")

        required_files = [
            self.train_dir,
            self.validation_dir,
            self.test_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, prefix='train')
        valid = self.process_dir(self.validation_dir, is_valid=True, prefix='valid')
        test = self.process_dir(self.test_dir, prefix='test')

        train.extend(valid)
        train.extend(test)

        super().__init__(train, [], [], **kwargs)

    def process_dir(self, dir_path, is_valid=False, prefix='train'):
        data = []
        if not is_valid:
            for dir_name in os.listdir(dir_path):
                img_lists = glob.glob(os.path.join(dir_path, dir_name, "*.jpg"))
                for img_path in img_lists:
                    pid = self.dataset_name + "_" + prefix + dir_name.split('-')[-1]
                    camid = self.dataset_name + "_" + "0"
                    data.append([img_path, pid, camid])
        else:
            query_path = os.path.join(dir_path, 'image_query')
            gallery_path = os.path.join(dir_path, 'image_gallery')
            query_file = os.path.join(dir_path, 'index_validation_query.txt')
            gallery_file = os.path.join(dir_path, 'index_validation_gallery.txt')

            with open(query_file, 'r') as f:
                query_list = [line.strip() for line in f.readlines()]
            for data_info in query_list:
                img_name, camid, pid, num_img = data_info.split(' ')
                img_path = os.path.join(query_path, img_name)
                pid = self.dataset_name + "_" + prefix + pid
                camid = self.dataset_name + "_" + camid
                data.append([img_path, pid, camid])

            with open(gallery_file, 'r') as f:
                gallery_list = [line.strip() for line in f.readlines()]

            for data_info in gallery_list:
                img_name, camid, pid, num_img = data_info.split(' ')
                img_path = os.path.join(gallery_path, img_name)
                pid = self.dataset_name + "_" + prefix + pid
                camid = self.dataset_name + "_" + camid
                data.append([img_path, pid, camid])
        return data
