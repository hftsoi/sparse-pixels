# ruff: noqa: E402
import matplotlib

matplotlib.use("Agg")  # headless: plotting utils must not open a window during tests

import keras
import numpy as np
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.layers import QConv2D, QDense
from hgq.quantizer.config import QuantizerConfig
from keras.layers import Activation, AveragePooling2D, Flatten

from sparsepixels.layers import AveragePooling2DSparse, InputReduce, QConv2DSparse
from sparsepixels.utils import SparseTrainingMonitor, plot_history, set_sparse_ebops_factor


def build_cnn(is_sparse, n=20, threshold=0.0, beta_n=1e-4, learn_n=True, learn_threshold=True):
    iq_conf = QuantizerConfig(place='datalane', q_type='kif', i0=4, f0=8, overflow_mode='WRAP')
    with (
        QuantizerConfigScope(place='all', default_q_type='kbi', overflow_mode='SAT_SYM'),
        QuantizerConfigScope(place='datalane', default_q_type='kif', overflow_mode='WRAP'),
        LayerConfigScope(enable_ebops=True, enable_iq=True, beta0=1e-5),
    ):
        x_in = keras.Input(shape=(32, 32, 1), name='x_in')
        if is_sparse:
            x, keep_mask = InputReduce(n=n, threshold=threshold, beta_n=beta_n,
                                       learn_n=learn_n, learn_threshold=learn_threshold,
                                       name='input_reduce')(x_in)
            x = QConv2DSparse(filters=1, kernel_size=7, name='conv1', padding='same', strides=1,
                              activation='relu', iq_conf=iq_conf)([x, keep_mask])
            x, keep_mask = AveragePooling2DSparse(4, name='pool1')([x, keep_mask])
            x = QConv2DSparse(filters=3, kernel_size=5, name='conv2', padding='same', strides=1,
                              activation='relu', iq_conf=iq_conf)([x, keep_mask])
            x, keep_mask = AveragePooling2DSparse(2, name='pool2')([x, keep_mask])
        else:
            x = QConv2D(filters=1, kernel_size=7, name='conv1', padding='same', strides=1,
                        activation='relu', iq_conf=iq_conf)(x_in)
            x = AveragePooling2D(4, name='pool1')(x)
            x = QConv2D(filters=3, kernel_size=5, name='conv2', padding='same', strides=1,
                        activation='relu', iq_conf=iq_conf)(x)
            x = AveragePooling2D(2, name='pool2')(x)

        x = Flatten(name='flatten')(x)
        x = QDense(36, name='dense1', activation='relu', iq_conf=iq_conf)(x)
        x = QDense(10, name='dense2', iq_conf=iq_conf)(x)
        x = Activation('softmax', name='softmax')(x)

    model = keras.Model(x_in, x, name='cnn_sparse' if is_sparse else 'cnn_full')
    if is_sparse:
        set_sparse_ebops_factor(model)  # sparse-correct the EBOPS (reporting + regularizer)
    return model


def test_build_full_cnn():
    m = build_cnn(is_sparse=False)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    m.summary()


def test_build_sparse_cnn():
    m = build_cnn(is_sparse=True, n=20)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    m.summary()
    ir = m.get_layer('input_reduce')
    assert isinstance(ir.n_max_pixels, int) and ir.n_max_pixels == 20
    assert isinstance(ir.threshold, float)


def test_sparse_cnn_fixed_mode():
    # both flags off is the plain, non-learnable selection; it should still build and compile
    m = build_cnn(is_sparse=True, n=15, threshold=0.2, learn_n=False, learn_threshold=False)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    assert m.get_layer('input_reduce').n_max_pixels == 15


def test_sparse_cnn_trains_with_monitor():
    # exercise the learnable path, EBOPS factor, monitor and plot on a tiny synthetic set
    rng = np.random.default_rng(0)
    x = rng.random((64, 32, 32, 1)).astype('float32')
    y = keras.utils.to_categorical(np.arange(64) % 10, 10)

    m = build_cnn(is_sparse=True, n=20, threshold=0.3, beta_n=1e-3)
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    history = m.fit(x, y, epochs=2, batch_size=32, verbose=0, callbacks=[SparseTrainingMonitor()])

    for key in ('loss', 'loss_task', 'loss_ebops', 'loss_n', 'loss_maskedE', 'n_max_pixels', 'threshold', 'ebops'):
        assert key in history.history
    plot_history(history)  # must run headless without error
