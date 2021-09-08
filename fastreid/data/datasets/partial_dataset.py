# encoding: utf-8

"""
@author:  lingxiao he
@contact: helingxiao3@jd.com
"""

import glob
import os
import os.path as osp
import re
import pdb
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['PartialREID', 'PartialiLIDS', 'OccludedREID', 'CrowdREID']


def process_test(query_path, gallery_path):
    query_img_paths = glob.glob(os.path.join(query_path, '*.jpg'))
    gallery_img_paths = glob.glob(os.path.join(gallery_path, '*.jpg'))
    query_paths = []
    pattern = re.compile(r'([-\d]+)_(\d*)')
    for img_path in query_img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        query_paths.append([img_path, pid, camid])
    gallery_paths = []
    for img_path in gallery_img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        gallery_paths.append([img_path, pid, camid])
    return query_paths, gallery_paths

def process_dir(query_path, gallery_path):
    query_img_paths = glob.glob(os.path.join(query_path, '*.jpg'))
    #pdb.set_trace()
    gallery_img_paths = glob.glob(os.path.join(gallery_path, '*.jpg'))
    query_paths = []
    pattern = re.compile(r'([-\d]+)_c(\d)')
    for img_path in query_img_paths:
        
        pid, camid = map(int, pattern.search(img_path).groups())
        query_paths.append([img_path, pid, camid])
    gallery_paths = []
    for img_path in gallery_img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        gallery_paths.append([img_path, pid, camid])
    return query_paths, gallery_paths
       

@DATASET_REGISTRY.register()
class PartialREID(ImageDataset):

    dataset_name = "partialreid"

    def __init__(self, root='datasets',):
        self.root = root

        self.query_dir = osp.join(self.root, 'Partial_REID/partial_body_images')
        self.gallery_dir = osp.join(self.root, 'Partial_REID/whole_body_images')
        query, gallery = process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery)


@DATASET_REGISTRY.register()
class PartialiLIDS(ImageDataset):
    dataset_name = "partialilids"

    def __init__(self, root='datasets',):
        self.root = root

        self.query_dir = osp.join(self.root, 'PartialiLIDS/query')
        self.gallery_dir = osp.join(self.root, 'PartialiLIDS/gallery')
        query, gallery = process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery)


@DATASET_REGISTRY.register()
class OccludedREID(ImageDataset):
    dataset_name = "occludereid"

    def __init__(self, root='datasets',):
        self.root = root

        self.query_dir = osp.join(self.root, 'OccludedREID/query')
        self.gallery_dir = osp.join(self.root, 'OccludedREID/gallery')
        query, gallery = process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery)

@DATASET_REGISTRY.register()
class CrowdREID(ImageDataset):
    dataset_name = "occludereid"

    def __init__(self, root='datasets',):
        self.root = root

        self.query_dir = osp.join(self.root, 'Crowd_REID/query')
        self.gallery_dir = osp.join(self.root, 'Crowd_REID/gallery')
        query, gallery = process_dir(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery)
