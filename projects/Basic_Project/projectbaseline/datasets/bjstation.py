# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import os.path as osp
import re

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['bjzStation', 'bjzExit', 'bjzEntrance', 'bjzCrowd', 'bjzBlack']


@DATASET_REGISTRY.register()
class bjzStation(ImageDataset):
    dataset_name = "bjz"
    dataset_dir = "beijingStation"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        train_list = [
            "train/train_summer",
            # 'train/train_winter',
            "train/train_summer_extra",
            # 'train/train_winter_20191204',
            # 'train/train_winter_20200102',
            # 'train/train_winter_20200329',
            # 'train/train_winter_20200618',
            "benchmark/train_bench_slim",
        ]
        self.train_path_list = [osp.join(self.dataset_dir, train_name) for train_name in train_list]

        required_files = self.train_path_list
        self.check_before_run(required_files)

        train = []
        for train_path in self.train_path_list:
            train.extend(self.process_train(train_path))

        # train = self.sample_subset(train, k=2)
        super().__init__(train=train, query=[], gallery=[])

    def process_train(self, dir_path):
        pattern = re.compile(r'([-\d]+)_c(\d*)')
        v_paths = []
        for d in os.listdir(dir_path):
            img_lists = glob.glob(os.path.join(dir_path, d, "*.jpg"))
            for img_path in img_lists:
                _, camid = map(str, pattern.search(img_path).groups())
                pid = self.dataset_name + "_" + d
                camid = self.dataset_name + "_" + camid
                v_paths.append([img_path, pid, camid])

        return v_paths

    def process_test(self, query_path, gallery_path):
        query_img_paths = []
        for root, _, files in os.walk(query_path):
            for file in files:
                if file.endswith('.jpg'):
                    query_img_paths.append(os.path.join(root, file))

        gallery_img_paths = []
        for root, _, files in os.walk(gallery_path):
            for file in files:
                if file.endswith('.jpg'):
                    gallery_img_paths.append(os.path.join(root, file))

        pattern = re.compile(r'([-\d]+)_c(\d*)')
        query_paths = []
        for img_path in query_img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            query_paths.append([img_path, pid, camid])

        gallery_paths = []
        for img_path in gallery_img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            gallery_paths.append([img_path, pid, camid])

        return query_paths, gallery_paths

    def sample_subset(self, samples, k=9999, seed=1234):
        import random
        random.seed(seed)

        v = {}
        for img_path, pid, camid in samples:
            if pid not in v.keys():
                v[pid] = {}
            if camid not in v[pid].keys():
                v[pid][camid] = []
            v[pid][camid].append([img_path, pid, camid])

        results = []
        for pid_vals in v.values():
            for cam_vals in pid_vals.values():
                results.extend(random.sample(cam_vals, k=min(k, len(cam_vals))))

        return results


@DATASET_REGISTRY.register()
class bjzExit(bjzStation, ImageDataset):
    dataset_dir = "beijingStation/benchmark"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.query_dir = osp.join(self.dataset_dir, 'exit_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'exit_gallery')

        required_files = [self.query_dir, self.gallery_dir]
        self.check_before_run(required_files)

        query, gallery = self.process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, train=[], query=query, gallery=gallery, **kwargs)


@DATASET_REGISTRY.register()
class bjzEntrance(bjzStation, ImageDataset):
    dataset_dir = "beijingStation/benchmark"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.query_dir = osp.join(self.dataset_dir, 'entrance_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'entrance_gallery')

        required_files = [self.query_dir, self.gallery_dir]
        self.check_before_run(required_files)

        query, gallery = self.process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery, **kwargs)


@DATASET_REGISTRY.register()
class bjzCrowd(bjzStation, ImageDataset):
    dataset_name = 'bjzcrowd'

    def __init__(self, root='datasets', ):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.query_dir = osp.join(self.dataset_dir, 'benchmark/Crowd_REID/Query')
        self.gallery_dir = osp.join(self.dataset_dir, 'benchmark/Crowd_REID/Gallery')

        query, gallery = self.process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery)


@DATASET_REGISTRY.register()
class bjzBlack(bjzStation, ImageDataset):
    dataset_name = 'bjzblack'

    def __init__(self, root='datasets', ):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.query_dir = osp.join(self.dataset_dir, 'benchmark/black_general_reid/query')
        self.gallery_dir = osp.join(self.dataset_dir, 'benchmark/black_general_reid/gallery')

        query, gallery = self.process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery)
