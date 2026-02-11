from sparsepixels.layers import *
from qkeras import *
from tensorflow.keras.layers import *

def build_cnn(is_sparse, B=16, I=6, n_max_pixels=None):
    quantizer = quantized_bits(B, I, alpha=1)
    quantized_relu = f'quantized_relu({B}, {I})'

    x_in = keras.Input(shape=(32, 32, 1), name='x_in')
    if is_sparse:
        x, keep_mask = InputReduce(n_max_pixels=n_max_pixels, threshold=1, name='input_reduce')(x_in)
    else:
        x = x_in

    if is_sparse:
        x = QConv2DSparse(filters=1, kernel_size=7, use_bias=True, name='conv1', padding='same', strides=1,
                          kernel_quantizer=quantizer, bias_quantizer=quantizer)([x, keep_mask])
        x = QActivation(quantized_relu, name='relu1')(x)
        x, keep_mask = AveragePooling2DSparse(4, name='pool1')([x, keep_mask])

        x = QConv2DSparse(filters=3, kernel_size=5, use_bias=True, name='conv2', padding='same', strides=1,
                          kernel_quantizer=quantizer, bias_quantizer=quantizer)([x, keep_mask])
        x = QActivation(quantized_relu, name='relu2')(x)
        x, keep_mask = AveragePooling2DSparse(2, name='pool2')([x, keep_mask])

    else:
        x = QConv2D(filters=1, kernel_size=7, use_bias=True, name='conv1', padding='same', strides=1,
                    kernel_quantizer=quantizer, bias_quantizer=quantizer)(x)
        x = QActivation(quantized_relu, name='relu1')(x)
        x = AveragePooling2D(4, name='pool1')(x)
        
        x = QConv2D(filters=3, kernel_size=5, use_bias=True, name='conv2', padding='same', strides=1,
                    kernel_quantizer=quantizer, bias_quantizer=quantizer)(x)
        x = QActivation(quantized_relu, name='relu2')(x)
        x = AveragePooling2D(2, name='pool2')(x)

    x = Flatten(name='flatten')(x)

    x = QDense(36, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='dense1')(x)
    x = QActivation(quantized_relu, name='relu3')(x)

    x = QDense(10, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='dense2')(x)
    x = Activation('softmax', name='softmax')(x)

    return keras.Model(x_in, x)


def test_build_full_cnn():
    cnn_full = build_cnn(is_sparse=False)
    cnn_full.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics = ['accuracy'])
    cnn_full.summary()

def test_build_sparse_cnn():
    cnn_sparse = build_cnn(is_sparse=True, n_max_pixels=20)
    cnn_sparse.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics = ['accuracy'])
    cnn_sparse.summary()