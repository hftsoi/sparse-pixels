<p align="center">
  <img src="https://raw.githubusercontent.com/hftsoi/sparse-pixels/main/docs/figs/logo.png" width="300" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/hftsoi/sparse-pixels/main/docs/figs/sparsepixels.png" width="900"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/hftsoi/sparse-pixels/main/docs/figs/cnn_standard.gif" width="400" />
  <img src="https://raw.githubusercontent.com/hftsoi/sparse-pixels/main/docs/figs/cnn_sparse.gif" width="400" />
</p>

# SparsePixels: Efficient convolution for sparse data on FPGAs

[![arXiv](https://img.shields.io/badge/arXiv-2512.06208-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2512.06208)
[![PyPI - Version](https://img.shields.io/pypi/v/sparsepixels?color=orange&style=flat-square)](https://pypi.org/project/sparsepixels)

> **Note:** we are actively working on integrating into hls4ml (first qkeras with keras2, and then HGQ with keras3), we are also working on a major upgrade with partial paralleliztion and streaming for sparse layers in HLS. stay tuned!!

## Installation
```
pip install sparsepixels
```

## Getting Started
On the model training in Python, import sparse layers:
```
from sparsepixels.layers import *
```
Sparse input reduction:
```
x_in = keras.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), name='x_in')
x, keep_mask = InputReduce(n_max_pixels=n_max_pixels, threshold=threshold, name='input_reduce')(x_in)
```
Sparse convolution:
```
x = QConv2DSparse(filters=1, kernel_size=7, use_bias=True, name='conv1', padding='same', strides=1,
                          kernel_quantizer=quantizer, bias_quantizer=quantizer)([x, keep_mask])
```
Sparse pooling:
```
x, keep_mask = AveragePooling2DSparse(4, name='pool1')([x, keep_mask])
```
We are working on hls4ml integration that auto parses the sparse layers into HLS.

## Documentation

## Citation
If you find this useful in your research, please consider citing:
```
@article{Tsoi:2025nvg,
    author = "Tsoi, Ho Fung and Rankin, Dylan and Loncar, Vladimir and Harris, Philip",
    title = "{SparsePixels: Efficient Convolution for Sparse Data on FPGAs}",
    eprint = "2512.06208",
    archivePrefix = "arXiv",
    primaryClass = "cs.AR",
    month = "12",
    year = "2025"
}
```
