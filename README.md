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

[![Docs](https://img.shields.io/badge/docs-site-informational?color=blue&style=flat-square)](https://hftsoi.github.io/sparse-pixels)
[![arXiv](https://img.shields.io/badge/arXiv-2512.06208-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2512.06208)
[![PyPI - Version](https://img.shields.io/pypi/v/sparsepixels?color=orange&style=flat-square)](https://pypi.org/project/sparsepixels)

SparsePixels is a Keras 3 library to build, train, and deploy sparse convolutional neural networks on FPGAs. In many detectors, especially in high-energy physics experiments, the images are almost empty: only a handful of pixels carry a signal (the hits), yet a standard CNN still spends compute on every pixel. A sparse CNN convolves only over the active pixels, so its cost scales with the number of hits rather than the image size, which is what makes low-latency, real-time inference (e.g. in a trigger) feasible on an FPGA. This library builds quantization-aware (with [HGQ2](https://github.com/calad0i/HGQ2) as a backend) sparse CNNs in which the pixel budget and the activity threshold can be learned from data, with a hardware-aware penalty that drives the budget toward the fewest pixels the task tolerates. Trained models convert to FPGA firmware through the [hls4ml](https://github.com/fastmachinelearning/hls4ml) integration, with control over the parallelization of the sparse layers so user can trade latency against resource usage. This training library is designed to mirror our HLS design with bit-exact accuracy.

## Installation

With Python >= 3.10:

```
pip install sparsepixels
```

This includes all necessary backends (tensorflow, keras, HGQ2).

## Getting Started

First, study the data to pick a threshold and an initial pixel budget `n`: how many pixels stay
active as the threshold rises, and what a candidate `(n, threshold)` maskes out pixels on a few example images.

```python
from sparsepixels.utils import active_pixels_vs_threshold, plot_reduced_examples

active_pixels_vs_threshold(x_train)
plot_reduced_examples(x_train, n=20, threshold=0.1, n_examples=3)
```

Build an example sparse CNN within HGQ2 quantization scopes. Set generous initial bit-widths at the
scope level: sparse signals are small (and get diluted further by pooling), so the low HGQ2 defaults
(4-bit weights, 2 fractional bits on activations) can quantize them, or entire low-magnitude
kernels, to exact zeros at initialization and block training. Starting wide (e.g. `b0=8` for
weights, `f0=8` for activations) keeps everything alive, and the EBOPS regularizer trims the
bit-widths back down during training. `InputReduce` keeps the first `n` active pixels (based on the
first channel above `threshold`) in a row-major order.
By default `n` and `threshold` are trainable hyperparameters: `beta_n` controls how aggressively `n` is driven smaller for a lower hardware footprint, and `beta_maskedE` penalizes over-masking pixel intensity.

```python
import keras
from hgq.layers import QDense
from hgq.config import QuantizerConfigScope, LayerConfigScope
from sparsepixels.layers import InputReduce, QConv2DSparse, AveragePooling2DSparse, MaxPooling2DSparse

with (
    QuantizerConfigScope(place='all', default_q_type='kbi', overflow_mode='SAT_SYM', b0=8, i0=0),
    QuantizerConfigScope(place='datalane', default_q_type='kif', overflow_mode='WRAP', i0=4, f0=8),
    LayerConfigScope(enable_ebops=True, enable_iq=True, beta0=1e-5),
):
    x_in = keras.Input(shape=(28, 28, 1), name='x_in')

    x, keep_mask = InputReduce(
        n=30,                  # initial pixel budget
        threshold=0.1,         # initial activity threshold
        beta_n=5e-3,           # higher -> drives the pixel budget n smaller
        beta_maskedE=1.0,      # higher -> prevents over-masking
        learn_n=True,          # trainable pixel budget
        learn_threshold=True,  # trainable threshold
        name='input_reduce',
    )(x_in)
    x = QConv2DSparse(filters=3, kernel_size=3, name='conv1', padding='same', strides=1,
                      activation='relu')([x, keep_mask])
    x, keep_mask = AveragePooling2DSparse(2, name='pool1')([x, keep_mask])

    x = keras.layers.Flatten(name='flatten')(x)
    x = QDense(10, name='dense1', activation='relu')(x)
    x = keras.layers.Activation('softmax', name='softmax')(x)

model = keras.Model(x_in, x)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
```

Train the model with `SparseTrainingMonitor`. It records the loss breakdown, the learned
budget/threshold, the EBOPS and the masked-intensity penalty each epoch, and it corrects the EBOPS
(a proxy for the quantized hardware cost) to the sparse compute automatically.

```python
from sparsepixels.utils import SparseTrainingMonitor

early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                                           patience=20, restore_best_weights=True)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=100, batch_size=128,
                    callbacks=[early_stop, SparseTrainingMonitor()])
```

After training, plot the diagnostics from the monitoring tool. The final pixel budget and threshold values are stored in `InputReduce`: `layer.n_max_pixels` and `layer.threshold` (hls4ml converter will auto-parse these from the model).

```python
from sparsepixels.utils import plot_history, print_quantization, plot_quantization

plot_history(history, early_stopping=early_stop)  # loss breakdown, budget, threshold, EBOPS
print_quantization(model)                         # per-layer bit-width distribution and EBOPS
plot_quantization(model)

ir = model.get_layer('input_reduce')
print(f"n_max_pixels={ir.n_max_pixels}, threshold={ir.threshold:.3f}")
```

## Converting a trained model to HLS with hls4ml

> **Note:** `sparsepixels` support is part of the official [hls4ml](https://github.com/fastmachinelearning/hls4ml) repo ([PR #1468](https://github.com/fastmachinelearning/hls4ml/pull/1468), merged). It is not in a tagged release yet, so install hls4ml from the main branch:
>
> ```bash
> pip install "git+https://github.com/fastmachinelearning/hls4ml.git"
> ```

Once installed, pull a config from the trained model, optionally set the per-layer parallelization
knobs, and convert:

```python
import hls4ml

hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
hls_config.setdefault('Model', {})['PipelineStyle'] = 'dataflow'  # "#pragma HLS DATAFLOW"

n_max_pixels = model.get_layer('input_reduce').n_max_pixels

# input reduce: 'tree' (default, lowest latency) or 'stream' (sequential, fewer resources)
hls_config['LayerName']['input_reduce']['Variant'] = 'tree'
# conv: active pixels in parallel (<= n_max_pixels), and filters in parallel (<= that conv's filters)
hls_config['LayerName']['conv1']['PixelParallelFactor'] = n_max_pixels
hls_config['LayerName']['conv1']['FiltParallelFactor'] = 3
# pool: active pixels in parallel, and channels in parallel
hls_config['LayerName']['pool1']['PixelParallelFactor'] = n_max_pixels
hls_config['LayerName']['pool1']['ChanParallelFactor'] = 3
# flatten: scatter positions in parallel (<= out_height * out_width; here 28*28 pools to 14*14)
hls_config['LayerName']['flatten']['ParallelFactor'] = 14 * 14

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=hls_config,
    output_dir='hls_proj/my_sparse_cnn',
    backend='Vitis',
    io_type='io_parallel',
)
hls_model.write()
hls_model.compile()
y_hls = hls_model.predict(x_test)
```

## Documentation

See [here](https://hftsoi.github.io/sparse-pixels/) for documentation with walk-through guidance.

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
