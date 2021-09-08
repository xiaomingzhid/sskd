# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import copy
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from fastreid.evaluation import ReidEvaluator
from fastreid.evaluation.query_expansion import aqe
from fastreid.evaluation.rank import evaluate_rank
from fastreid.evaluation.rerank import re_ranking
from fastreid.evaluation.roc import evaluate_roc
from fastreid.utils import comm

logger = logging.getLogger("fastreid.cam_evaluator")


class CamEvaluator(ReidEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        super().__init__(cfg, num_query, output_dir=output_dir)

        # Add camera features
        self.cam_features = []

    def reset(self):
        self.features = []
        self.cam_features = []
        self.pids = []
        self.camids = []

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.features.append(outputs[0].cpu())
        self.cam_features.append(outputs[1].cpu())


def evaluate(self):
    if comm.get_world_size() > 1:
        comm.synchronize()
        features = comm.gather(self.features)
        features = sum(features, [])

        cam_features = comm.gather(self.cam_features)
        cam_features = sum(cam_features, [])

        pids = comm.gather(self.labels)
        pids = sum(pids, [])

        camids = comm.gather(self.camids)
        camids = sum(camids, [])

        if not comm.is_main_process():
            return {}
    else:
        features = self.features
        cam_features = self.cam_features
        pids = self.labels
        camids = self.camids

    features = torch.cat(features, dim=0)
    cam_features = torch.cat(cam_features, dim=0)

    # query feature, person ids and camera ids
    query_features = features[:self._num_query]
    query_cam_features = cam_features[:self._num_query]
    query_pids = np.asarray(pids[:self._num_query])
    query_camids = np.asarray(camids[:self._num_query])

    # gallery features, person ids and camera ids
    gallery_features = features[self._num_query:]
    gallery_cam_features = cam_features[self._num_query:]
    gallery_pids = np.asarray(pids[self._num_query:])
    gallery_camids = np.asarray(camids[self._num_query:])

    self._results = OrderedDict()

    if self.cfg.TEST.AQE.ENABLED:
        logger.info("Test with AQE setting")
        qe_time = self.cfg.TEST.AQE.QE_TIME
        qe_k = self.cfg.TEST.AQE.QE_K
        alpha = self.cfg.TEST.AQE.ALPHA
        query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

    if self.cfg.TEST.METRIC == "cosine":
        query_features = F.normalize(query_features, dim=1)
        query_cam_features = F.normalize(query_cam_features, dim=1)
        gallery_features = F.normalize(gallery_features, dim=1)
        gallery_cam_features = F.normalize(gallery_cam_features, dim=1)

    dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)
    cam_dist = self.cal_dist(self.cfg.TEST.METRIC, query_cam_features, gallery_cam_features)
    dist = dist - 0.2 * cam_dist

    if self.cfg.TEST.RERANK.ENABLED:
        logger.info("Test with rerank setting")
        k1 = self.cfg.TEST.RERANK.K1
        k2 = self.cfg.TEST.RERANK.K2
        lambda_value = self.cfg.TEST.RERANK.LAMBDA
        q_q_dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, query_features)
        g_g_dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, gallery_features)
        re_dist = re_ranking(dist, q_q_dist, g_g_dist, k1, k2, lambda_value)
        query_features = query_features.numpy()
        gallery_features = gallery_features.numpy()
        cmc, all_AP, all_INP = evaluate_rank(re_dist, query_features, gallery_features,
                                             query_pids, gallery_pids, query_camids,
                                             gallery_camids, use_distmat=True)
    else:
        query_features = query_features.numpy()
        gallery_features = gallery_features.numpy()
        cmc, all_AP, all_INP = evaluate_rank(dist, query_features, gallery_features,
                                             query_pids, gallery_pids, query_camids, gallery_camids,
                                             use_distmat=True)
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        self._results['Rank-{}'.format(r)] = cmc[r - 1]
    self._results['mAP'] = mAP
    self._results['mINP'] = mINP

    if self.cfg.TEST.ROC_ENABLED:
        scores, labels = evaluate_roc(dist, query_features, gallery_features,
                                      query_pids, gallery_pids, query_camids, gallery_camids)
        fprs, tprs, thres = metrics.roc_curve(labels, scores)

        for fpr in [1e-4, 1e-3, 1e-2]:
            ind = np.argmin(np.abs(fprs - fpr))
            self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

    return copy.deepcopy(self._results)
