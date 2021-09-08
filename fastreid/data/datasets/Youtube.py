# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
import glob
import pdb
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['YouTube', ]


@DATASET_REGISTRY.register()
class YouTube(ImageDataset):
    dataset_dir = "YouTube_data"
    dataset_name = "YouTube"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []
        files= os.listdir(train_path)
        pid = 0
        camid = str(1)
        for imgfile in files:
            img_names = glob.glob(os.path.join(train_path, imgfile, "*.jpg"))
            
            for img_name in img_names:
                #pdb.set_trace()
                data.append([img_name, str(pid), camid])
            pid = pid + 1
            

        return data
