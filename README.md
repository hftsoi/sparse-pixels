





# SparsePixels: Efficient convolution for sparse data on FPGAs

[arXiv](https://arxiv.org/abs/2512.06208)
[PyPI - Version](https://pypi.org/project/sparsepixels)

> **Note:** We are actively working on hls4ml integration to auto-convert sparse models to HLS, along with a major upgrade with partial parallelization and streaming for sparse layers in HLS. Stay tuned!

## Installation

With Python >= 3.10:

```
pip install sparsepixels
```

## Getting Started

Import sparse layers and quantization library (HGQ2):

```python
import keras
from keras.layers import Flatten, Activation, ReLU
from hgq.layers import QConv2D, QDense
from hgq.config import QuantizerConfigScope, LayerConfigScope
from hgq.quantizer.config import QuantizerConfig
from sparsepixels.layers import InputReduce, QConv2DSparse, AveragePooling2DSparse
```

Build an example sparse CNN within HGQ2 quantization scopes:

```python
with (
    QuantizerConfigScope(place='all', default_q_type='kbi', overflow_mode='SAT_SYM'),
    QuantizerConfigScope(place='datalane', default_q_type='kif', overflow_mode='WRAP'),
    LayerConfigScope(enable_ebops=False, enable_iq=False),
):
    x_in = keras.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), name='x_in')

    # Sparse input reduction: retain up to n_max_pixels active pixels
    x, keep_mask = InputReduce(n_max_pixels=20, threshold=0.1, name='input_reduce')(x_in)

    # Sparse convolution
    x = QConv2DSparse(filters=3, kernel_size=3, name='conv1', padding='same', strides=1,
                      bq_conf=QuantizerConfig('default', 'bias'))([x, keep_mask])
    x = ReLU(name='relu1')(x)

    # Sparse pooling
    x, keep_mask = AveragePooling2DSparse(2, name='pool1')([x, keep_mask])

    x = Flatten(name='flatten')(x)
    x = QDense(10, name='dense1', activation='relu')(x)
    x = Activation('softmax', name='softmax')(x)

model = keras.Model(x_in, x)
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

