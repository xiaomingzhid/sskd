# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import sys
import os
import os.path as osp
import pdb
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
##### Log #####
# 22.01.2019
# - add v2
# - v1 and v2 differ in dir names
# - note that faces in v2 are blurred
TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'MSMT17_V1': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    }
}


@DATASET_REGISTRY.register()
class TestSet(ImageDataset):
    """MSMT17.
    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_

    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    # dataset_dir = 'MSMT17_V2'
    dataset_url = None
    dataset_name = 'TestSet'

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = root

        self.list_query_path = osp.join(self.dataset_dir, 'query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'gallery.txt')
        
        query = self.process_dir(self.dataset_dir, self.list_query_path, is_train=False)
        gallery = self.process_dir(self.dataset_dir, self.list_gallery_path, is_train=False)

        super(TestSet, self).__init__([], query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path, is_train=True):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        data = []
        

        for img_idx, img_info in enumerate(lines):
            #pdb.set_trace()
            img_path, pid, camid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            
            camid = int(camid) - 1  # index starts from 0
            img_path = osp.join(dir_path, img_path)
            
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
