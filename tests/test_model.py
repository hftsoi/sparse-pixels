import keras
from keras.layers import Flatten, Activation, AveragePooling2D, ReLU
from hgq.layers import QConv2D, QDense
from hgq.config import QuantizerConfigScope, LayerConfigScope
from sparsepixels.layers import InputReduce, QConv2DSparse, AveragePooling2DSparse


def build_cnn(is_sparse, n_max_pixels=None):
    with (
        QuantizerConfigScope(place='all', default_q_type='kbi', overflow_mode='SAT_SYM'),
        QuantizerConfigScope(place='datalane', default_q_type='kif', overflow_mode='WRAP'),
        LayerConfigScope(enable_ebops=False, enable_iq=False),
    ):
        x_in = keras.Input(shape=(32, 32, 1), name='x_in')
        if is_sparse:
            x, keep_mask = InputReduce(n_max_pixels=n_max_pixels, threshold=1, name='input_reduce')(x_in)
        else:
            x = x_in

        if is_sparse:
            x = QConv2DSparse(filters=1, kernel_size=7, name='conv1', padding='same', strides=1)([x, keep_mask])
            x = ReLU(name='relu1')(x)
            x, keep_mask = AveragePooling2DSparse(4, name='pool1')([x, keep_mask])

            x = QConv2DSparse(filters=3, kernel_size=5, name='conv2', padding='same', strides=1)([x, keep_mask])
            x = ReLU(name='relu2')(x)
            x, keep_mask = AveragePooling2DSparse(2, name='pool2')([x, keep_mask])
        else:
            x = QConv2D(filters=1, kernel_size=7, name='conv1', padding='same', strides=1,
                        activation='relu')(x)
            x = AveragePooling2D(4, name='pool1')(x)

            x = QConv2D(filters=3, kernel_size=5, name='conv2', padding='same', strides=1,
                        activation='relu')(x)
            x = AveragePooling2D(2, name='pool2')(x)

        x = Flatten(name='flatten')(x)

        x = QDense(36, name='dense1', activation='relu')(x)

        x = QDense(10, name='dense2')(x)
        x = Activation('softmax', name='softmax')(x)

    return keras.Model(x_in, x)


def test_build_full_cnn():
    cnn_full = build_cnn(is_sparse=False)
    cnn_full.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_full.summary()

def test_build_sparse_cnn():
    cnn_sparse = build_cnn(is_sparse=True, n_max_pixels=20)
    cnn_sparse.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_sparse.summary()
