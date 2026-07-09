# Quick Start

This page builds, trains, and reads out a sparse CNN end to end. For the same workflow on a toy dataset, see the [Sparse MNIST demo](demo/notebooks/sparse_mnist.ipynb).

## Study the data

First, pick a `threshold` and an initial pixel budget `n`: how many pixels stay active as the threshold rises, and what a candidate `(n, threshold)` keeps on a few example images.

``` python
from sparsepixels.utils import active_pixels_vs_threshold, plot_reduced_examples

active_pixels_vs_threshold(x_train)
plot_reduced_examples(x_train, n=20, threshold=0.1, n_examples=3)
```

## Build the model

Build a sparse CNN inside HGQ2 quantization scopes. `InputReduce` keeps the first `n` active pixels (first channel above `threshold`) in row-major order; by default `n` and `threshold` are trainable, with `beta_n` and `beta_maskedE` regularizing how they vary to avoid over-masking and under-masking.

``` python
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
```

!!! tip "Initial bit-widths"

    Start the quantizers wide at the scope level: `b0=8` for weights and `f0=8` for activations. Sparse signals are small and get diluted by pooling, so the lower HGQ2 defaults (4-bit weights, 2 fractional bits on activations) can quantize them, or entire low-magnitude kernels, to exact zeros at initialization, which blocks training from the start. The EBOPS regularizer trims the widths back down during training.

## Train

Add `SparseTrainingMonitor` to the callbacks. It records the loss breakdown, the learned budget/threshold and the EBOPS each epoch, and it sparse-corrects the EBOPS automatically. Pair it with an `EarlyStopping` that has `restore_best_weights=True`.

``` python
from sparsepixels.utils import SparseTrainingMonitor

early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=20, restore_best_weights=True)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy', metrics=['accuracy'],
)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=100, batch_size=128, callbacks=[early_stop, SparseTrainingMonitor()])
```

## Read out the results

After training, plot the diagnostics and read the values to deploy. The final pixel budget and threshold are stored on the `InputReduce` layer as `n_max_pixels` and `threshold` (the hls4ml converter auto-parses these from the model so you don't need to do anything).

``` python
from sparsepixels.utils import plot_history, print_quantization, plot_quantization

plot_history(history, early_stopping=early_stop)  # loss breakdown, budget, threshold, EBOPS
print_quantization(model)                         # per-layer bit-width distribution and EBOPS
plot_quantization(model)

ir = model.get_layer('input_reduce')
print(f"n_max_pixels={ir.n_max_pixels}, threshold={ir.threshold:.3f}")
```

## Next steps

- [Training & Monitoring](training.md): how to read the loss breakdown and tune `beta_n` / `beta_maskedE`.
- [HLS Conversion](conversion.md): turn the trained model into FPGA firmware.
