import tensorflow as tf
from qkeras import QConv2D, quantizers
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D

class InputReduce(tf.keras.layers.Layer):
    def __init__(self, n_max_pixels, threshold, **kwargs):
        super(InputReduce, self).__init__(**kwargs)
        self.n_max_pixels = n_max_pixels
        self.threshold = threshold

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]

        '''
        if self.threshold is not None:
            cond = inputs > self.threshold
        else:
            cond = inputs != 0
        active_flag = tf.cast(tf.reduce_any(cond, axis=-1), tf.int32)
        '''

        # to be consistent with hls, check only the first input channel
        if self.threshold is not None:
            active_flag = tf.cast(inputs[..., 0] > self.threshold, tf.int32)
        else:
            active_flag = tf.cast(inputs[..., 0] != 0, tf.int32)

        active_flag_flat = tf.reshape(active_flag, [batch_size, h * w])
        active_count = tf.cumsum(active_flag_flat, axis=1)

        keep_mask_flat = tf.cast(tf.logical_and(active_flag_flat == 1, active_count <= self.n_max_pixels), inputs.dtype)
        keep_mask = tf.reshape(keep_mask_flat, [batch_size, h, w, 1])

        inputs_reduced = inputs * keep_mask
        return inputs_reduced, keep_mask

    def get_config(self):
        config = super(InputReduce, self).get_config()
        config.update({
            "n_max_pixels": self.n_max_pixels,
            "threshold": self.threshold
        })
        return config
    

class RemoveDilatedPixels(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RemoveDilatedPixels, self).__init__(**kwargs)

    def call(self, inputs):
        x, mask = inputs
        mask = tf.cast(mask, x.dtype)
        removed = x * mask
        return removed

    def get_config(self):
        config = super(RemoveDilatedPixels, self).get_config()
        return config


class QConv2DSparse(tf.keras.layers.Layer):
    def __init__(self, *conv_args, **conv_kwargs):
        super().__init__(name=conv_kwargs.get("name", None))
        self._bias_quant_cfg = conv_kwargs.pop("bias_quantizer", None)
        self._bias_quantizer = (quantizers.get_quantizer(self._bias_quant_cfg) if self._bias_quant_cfg is not None else None)

        conv_kwargs["use_bias"] = False
        self.conv = QConv2D(*conv_args, **conv_kwargs)
        self.bias = self.add_weight(
            name = "bias",
            shape = (self.conv.filters,),
            initializer = "zeros",
            trainable = True,
            dtype = self.conv.dtype,
        )
        self.masker = RemoveDilatedPixels()

    def call(self, inputs, **kwargs):
        x, keep_mask = inputs
        x = self.masker((x, keep_mask))
        y = self.conv(x, **kwargs)

        b = self.bias
        if self._bias_quantizer is not None:
            b = self._bias_quantizer(b)
        b = tf.reshape(b, shape=(1, 1, 1, -1))

        non_zero = tf.cast(tf.not_equal(y, 0), y.dtype)
        y = y + b * non_zero

        y = self.masker((y, keep_mask))
        return y

    def get_config(self):
        cfg = super().get_config()
        cfg["conv_config"] = self.conv.get_config()
        cfg["bias_quantizer"] = self._bias_quant_cfg
        return cfg

    @classmethod
    def from_config(cls, config):
        conv_cfg = config.pop("conv_config")
        bias_quant_cfg = config.pop("bias_quantizer", None)
        layer = cls(**conv_cfg, bias_quantizer=bias_quant_cfg)
        return layer


class AveragePooling2DSparse(tf.keras.layers.Layer):
    def __init__(self, *pool_args, **pool_kwargs):
        super().__init__(name=pool_kwargs.get("name", None))
        self.avg_pool = AveragePooling2D(*pool_args, **pool_kwargs)
        self.max_pool = MaxPooling2D(*pool_args, **pool_kwargs)

    def call(self, inputs, **kwargs):
        x, keep_mask = inputs
        y = self.avg_pool(x, **kwargs)
        keep_mask_pooled = self.max_pool(keep_mask)
        return y, keep_mask_pooled

    def get_config(self):
        cfg = super().get_config()
        cfg["pool_config"] = self.avg_pool.get_config()
        return cfg

    @classmethod
    def from_config(cls, config):
        pool_cfg = config.pop("pool_config")
        return cls(**pool_cfg)
    

class MaxPooling2DSparse(tf.keras.layers.Layer):
    def __init__(self, *pool_args, **pool_kwargs):
        super().__init__(name=pool_kwargs.get("name", None))
        self.max_pool = MaxPooling2D(*pool_args, **pool_kwargs)

    def call(self, inputs, **kwargs):
        x, keep_mask = inputs
        y = self.max_pool(x, **kwargs)
        keep_mask_pooled = self.max_pool(keep_mask)
        return y, keep_mask_pooled

    def get_config(self):
        cfg = super().get_config()
        cfg["pool_config"] = self.max_pool.get_config()
        return cfg

    @classmethod
    def from_config(cls, config):
        pool_cfg = config.pop("pool_config")
        return cls(**pool_cfg)

