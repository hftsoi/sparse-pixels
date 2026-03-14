import keras
from hgq.layers import QConv2D
from hgq.quantizer import Quantizer
from hgq.quantizer.config import QuantizerConfig
from keras import ops
from keras.layers import AveragePooling2D, MaxPooling2D


class InputReduce(keras.layers.Layer):
    def __init__(self, n_max_pixels, threshold, **kwargs):
        super().__init__(**kwargs)
        self.n_max_pixels = n_max_pixels
        self.threshold = threshold

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        h = ops.shape(inputs)[1]
        w = ops.shape(inputs)[2]

        # to be consistent with hls, check only the first input channel
        if self.threshold is not None:
            active_flag = ops.cast(inputs[..., 0] > self.threshold, "int32")
        else:
            active_flag = ops.cast(inputs[..., 0] != 0, "int32")

        active_flag_flat = ops.reshape(active_flag, [batch_size, h * w])
        active_count = ops.cumsum(active_flag_flat, axis=1)

        keep_mask_flat = ops.cast(
            ops.logical_and(active_flag_flat == 1, active_count <= self.n_max_pixels),
            inputs.dtype,
        )
        keep_mask = ops.reshape(keep_mask_flat, [batch_size, h, w, 1])

        inputs_reduced = inputs * keep_mask
        return inputs_reduced, keep_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_max_pixels": self.n_max_pixels,
                "threshold": self.threshold,
            }
        )
        return config


class RemoveDilatedPixels(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, mask = inputs
        mask = ops.cast(mask, x.dtype)
        return x * mask

    def get_config(self):
        return super().get_config()


class QConv2DSparse(keras.layers.Layer):
    def __init__(self, *conv_args, **conv_kwargs):
        super().__init__(name=conv_kwargs.get("name", None))
        self._use_bias = conv_kwargs.pop("use_bias", True)
        self._bq_conf = conv_kwargs.pop("bq_conf", None) or QuantizerConfig("default", "bias")

        conv_kwargs["use_bias"] = False
        conv_kwargs.setdefault("enable_iq", False)
        self.conv = QConv2D(*conv_args, **conv_kwargs)
        self.masker = RemoveDilatedPixels()

    def build(self, input_shape):
        if self._use_bias:
            self.sparse_bias = self.add_weight(
                name="sparse_bias",
                shape=(self.conv.filters,),
                initializer="zeros",
                trainable=True,
            )
            self._bq = Quantizer(self._bq_conf, name=f"{self.name}_bq")
            self._bq.build((self.conv.filters,))
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x, keep_mask = inputs
        x = self.masker((x, keep_mask))
        y = self.conv(x, **kwargs)

        if self._use_bias:
            b = self._bq(self.sparse_bias)
            b = ops.reshape(b, (1, 1, 1, -1))
            non_zero = ops.cast(y != 0, y.dtype)
            y = y + b * non_zero

        y = self.masker((y, keep_mask))
        return y

    def get_config(self):
        cfg = super().get_config()
        cfg["conv_config"] = self.conv.get_config()
        cfg["use_bias"] = self._use_bias
        cfg["bq_conf"] = self._bq_conf
        return cfg

    @classmethod
    def from_config(cls, config):
        conv_cfg = config.pop("conv_config")
        use_bias = config.pop("use_bias", True)
        bq_conf = config.pop("bq_conf", None)
        return cls(**conv_cfg, use_bias=use_bias, bq_conf=bq_conf)


class AveragePooling2DSparse(keras.layers.Layer):
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


class MaxPooling2DSparse(keras.layers.Layer):
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
