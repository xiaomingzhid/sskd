# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import os
import numpy as np

if __name__ == "__main__":
    feat_name = os.listdir("r50_ibn-onnx_feat")
    for name in feat_name:
        caffe_output = np.load(os.path.join("r50_ibn-onnx_feat", name))
        pytorch_output = np.load(os.path.join("r50_ibn-pth_feat", name))
        sim = np.dot(caffe_output, pytorch_output.transpose())
        print(sim)
        np.testing.assert_allclose(caffe_output, caffe_output, rtol=1e-3, atol=1e-6)
        print('all test passed!')
