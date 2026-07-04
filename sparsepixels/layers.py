import keras
from hgq.layers import QConv2D
from hgq.quantizer import Quantizer
from hgq.quantizer.config import QuantizerConfig
from keras import ops
from keras.layers import AveragePooling2D, MaxPooling2D


class _ClipToRange(keras.constraints.Constraint):
    """Weight constraint that clips values to the range [lo, hi] after each update."""

    def __init__(self, lo, hi):
        self.lo = float(lo)
        self.hi = float(hi)

    def __call__(self, w):
        return ops.clip(w, self.lo, self.hi)

    def get_config(self):
        return {"lo": self.lo, "hi": self.hi}


class InputReduce(keras.layers.Layer):
    """Reduce a dense image to its first n active pixels for sparse FPGA inference.

    Keeps the first n pixels whose first channel is above threshold, in raster order, and zeroes the
    rest, returning the masked image together with a 0/1 keep mask that the following sparse layers
    use as the sparse representation.

    The budget n and the threshold can be learned during training (the default) so they need not be
    tuned by hand; set learn_n or learn_threshold to False to keep either fixed (both False gives the
    plain, non-learnable selection). The selection is always exact -- the learnable versions only
    shape the gradient, so the layer behaves identically at inference and stays deployable. When n is
    learned, a penalty of weight beta_n nudges it smaller, trading a little accuracy for lower FPGA
    latency and resources; it starts at n and can range over [1, H*W]. An optional penalty of weight
    beta_maskedE discourages masking pixel intensity, giving the threshold (and n) a restoring force so a
    trainable threshold settles near the noise floor instead of over-masking signal. After training,
    read the values to deploy from the n_max_pixels and threshold properties.

    Args:
        n: initial pixel budget, and the fixed budget when learn_n is False.
        threshold: initial activity threshold on the first channel, fixed when learn_threshold is False.
        beta_n: weight of the budget penalty added to the loss (0 disables it).
        beta_maskedE: weight of the masked-intensity penalty (0 disables it). Penalizes the fraction of pixel
            intensity that gets masked, so masking bright (signal) pixels is costly while masking dim
            (noise) pixels is cheap -- an adaptive, non-vanishing restoring force against over-masking.
        learn_n: make the pixel budget trainable.
        learn_threshold: make the threshold trainable.
        tau_threshold: softness of the threshold surrogate used to obtain gradients.
        tau_n: softness of the budget-cutoff surrogate used to obtain gradients.
    """

    def __init__(
        self,
        n=30,
        threshold=0.0,
        beta_n=5e-3,
        beta_maskedE=1.0,
        learn_n=True,
        learn_threshold=True,
        tau_threshold=0.05,
        tau_n=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_init = int(n)
        self.threshold_init = float(threshold)
        self.beta_n = float(beta_n)
        self.beta_maskedE = float(beta_maskedE)
        self.learn_n = learn_n
        self.learn_threshold = learn_threshold
        self.tau_threshold = float(tau_threshold)
        self.tau_n = float(tau_n)

    def build(self, input_shape):
        if self.learn_threshold:
            self.threshold_w = self.add_weight(
                name="threshold",
                shape=(),
                initializer=keras.initializers.Constant(self.threshold_init),
                trainable=True,
                constraint=keras.constraints.NonNeg(),
            )
        if self.learn_n:
            # Parametrize the budget as a fraction of its initial value (n = n_init * n_frac) so it
            # moves at a useful rate under Adam; clip to [1, H*W] so it can shrink to a single pixel
            # or grow up to keeping every pixel.
            if input_shape[1] and input_shape[2]:
                hi = (input_shape[1] * input_shape[2]) / self.n_init
            else:
                hi = 4.0
            self.n_frac = self.add_weight(
                name="n_frac",
                shape=(),
                initializer=keras.initializers.Constant(1.0),
                trainable=True,
                constraint=_ClipToRange(1.0 / self.n_init, max(1.0, hi)),
            )
        if self.beta_maskedE > 0:
            # Running masked-intensity fraction, tracked (not trained) so SparseTrainingMonitor can
            # report the penalty; refreshed each call.
            self._masked_intensity = self.add_weight(
                name="masked_intensity", shape=(), initializer="zeros", trainable=False
            )
        super().build(input_shape)

    def call(self, inputs):
        dt = inputs.dtype
        batch_size = ops.shape(inputs)[0]
        h = ops.shape(inputs)[1]
        w = ops.shape(inputs)[2]
        score = ops.reshape(inputs[..., 0], [batch_size, h * w])

        thr = self.threshold_w if self.learn_threshold else ops.cast(self.threshold_init, dt)
        n = ops.cast(self.n_init, dt) * self.n_frac if self.learn_n else ops.cast(self.n_init, dt)

        # Exact selection used for the forward pass.
        active_hard = ops.cast(score > thr, dt)
        rank_hard = ops.cumsum(active_hard, axis=1)
        keep_hard = active_hard * ops.cast(rank_hard <= ops.round(n), dt)

        if self.learn_threshold or self.learn_n:
            # Differentiable surrogate used only for the gradient (straight-through to the exact
            # selection above); the score > 0 gate keeps zero background pixels out when threshold=0.
            active_soft = ops.sigmoid((score - thr) / self.tau_threshold) * ops.cast(score > 0, dt)
            rank_soft = ops.cumsum(active_soft, axis=1)
            keep_soft = active_soft * ops.sigmoid((n - rank_soft) / self.tau_n)
            keep_flat = keep_soft + ops.stop_gradient(keep_hard - keep_soft)

            if self.beta_maskedE > 0:
                # Fraction of total pixel intensity that gets masked, added straight to the loss (not
                # through the classifier) so its gradient does not vanish for the masked pixels.
                # Masking bright pixels costs more than masking dim ones, so this pushes the threshold
                # back down (and n up) before it eats into signal -- an adaptive anti-over-masking term.
                masked_frac = ops.mean(ops.sum(score * (1.0 - keep_soft), axis=1) / (ops.sum(score, axis=1) + 1e-6))
                self.add_loss(self.beta_maskedE * masked_frac)
                self._masked_intensity.assign(masked_frac)
        else:
            keep_flat = keep_hard

        keep_mask = ops.reshape(keep_flat, [batch_size, h, w, 1])
        inputs_reduced = inputs * keep_mask

        if self.learn_n:
            self.add_loss(self.beta_n * n)

        return inputs_reduced, keep_mask

    @property
    def n_max_pixels(self):
        """Integer pixel budget to deploy (the initial value until the layer is built)."""
        if self.learn_n and self.built:
            return int(round(self.n_init * float(ops.convert_to_numpy(self.n_frac))))
        return self.n_init

    @property
    def threshold(self):
        """Threshold to deploy (the initial value until the layer is built)."""
        if self.learn_threshold and self.built:
            return float(ops.convert_to_numpy(self.threshold_w))
        return self.threshold_init

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n": self.n_init,
                "threshold": self.threshold_init,
                "beta_n": self.beta_n,
                "beta_maskedE": self.beta_maskedE,
                "learn_n": self.learn_n,
                "learn_threshold": self.learn_threshold,
                "tau_threshold": self.tau_threshold,
                "tau_n": self.tau_n,
            }
        )
        return config


class RemoveDilatedPixels(keras.layers.Layer):
    """Re-apply the keep mask, zeroing every pixel that is not active.

    Multiplies a feature map by its 0/1 keep mask (broadcast over channels) so only the kept pixels
    carry values. Used inside the sparse layers to restore the sparse representation after a dense op.

    Call args:
        inputs: tuple (x, mask) of the feature map and its keep mask.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, mask = inputs
        mask = ops.cast(mask, x.dtype)
        return x * mask

    def get_config(self):
        return super().get_config()


class QConv2DSparse(keras.layers.Layer):
    """Quantized 2D convolution that operates on the sparse (active-pixel) representation.

    Wraps an HGQ QConv2D: masks the input to the active pixels, convolves, adds a separately
    quantized per-filter bias on the nonzero outputs, applies the activation, then re-masks the
    output. This is numerically the same as a dense quantized conv restricted to the active pixels,
    which is what the HLS sparse_conv kernel computes.

    Args:
        *conv_args: positional arguments forwarded to hgq.layers.QConv2D (e.g. filters, kernel_size).
        **conv_kwargs: keyword arguments forwarded to QConv2D (padding, strides, ...). use_bias,
            activation and bq_conf are handled here: the bias has its own weight and quantizer
            (bq_conf), and the activation is applied after the bias.

    Call args:
        inputs: tuple (x, keep_mask) of the feature map and its keep mask.
    """

    def __init__(self, *conv_args, **conv_kwargs):
        super().__init__(name=conv_kwargs.get("name", None))
        self._use_bias = conv_kwargs.pop("use_bias", True)
        self._bq_conf = conv_kwargs.pop("bq_conf", None) or QuantizerConfig("default", "bias")
        self._activation = keras.activations.get(conv_kwargs.pop("activation", None))

        conv_kwargs["use_bias"] = False
        conv_kwargs["activation"] = None
        self.conv = QConv2D(*conv_args, **conv_kwargs)
        self.masker = RemoveDilatedPixels()

    def build(self, input_shape):
        # Build the wrapped conv eagerly here rather than lazily in call(): building it while Keras
        # symbolically traces call() triggers an HGQ weight check that fails in graph mode.
        x_shape = input_shape[0]
        if not self.conv.built:
            self.conv.build(x_shape)
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

    def compute_output_shape(self, input_shape):
        # Return the shape directly so Keras does not trace call() (masking preserves the shape).
        return self.conv.compute_output_shape(input_shape[0])

    def call(self, inputs, **kwargs):
        x, keep_mask = inputs
        x = self.masker((x, keep_mask))
        y = self.conv(x, **kwargs)

        if self._use_bias:
            b = self._bq(self.sparse_bias)
            b = ops.reshape(b, (1, 1, 1, -1))
            non_zero = ops.cast(y != 0, y.dtype)
            y = y + b * non_zero

        if self._activation is not None:
            y = self._activation(y)

        y = self.masker((y, keep_mask))
        return y

    def get_config(self):
        cfg = super().get_config()
        cfg["conv_config"] = self.conv.get_config()
        cfg["use_bias"] = self._use_bias
        cfg["bq_conf"] = self._bq_conf
        cfg["activation"] = keras.activations.serialize(self._activation)
        return cfg

    @classmethod
    def from_config(cls, config):
        conv_cfg = config.pop("conv_config")
        use_bias = config.pop("use_bias", True)
        bq_conf = config.pop("bq_conf", None)
        activation = config.pop("activation", None)
        return cls(**conv_cfg, use_bias=use_bias, bq_conf=bq_conf, activation=activation)


class AveragePooling2DSparse(keras.layers.Layer):
    """Average pooling on the sparse representation.

    Average-pools the feature map and max-pools the keep mask, so a pooled cell stays active when any
    of its source pixels were active. Mirrors the HLS sparse_pooling_avg kernel.

    Args:
        *pool_args: positional arguments forwarded to keras AveragePooling2D (e.g. pool_size).
        **pool_kwargs: keyword arguments forwarded to the pooling layers.

    Call args:
        inputs: tuple (x, keep_mask) of the feature map and its keep mask.
    """

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
    """Max pooling on the sparse representation.

    Max-pools both the feature map and the keep mask. Mirrors the HLS sparse_pooling_max kernel.

    Args:
        *pool_args: positional arguments forwarded to keras MaxPooling2D (e.g. pool_size).
        **pool_kwargs: keyword arguments forwarded to the pooling layer.

    Call args:
        inputs: tuple (x, keep_mask) of the feature map and its keep mask.
    """

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
