"""Utilities for studying sparse-pixel data and monitoring sparse-model training.

Before training, active_pixels_vs_threshold and plot_reduced_examples help pick a threshold and an
initial pixel budget. During training, add SparseTrainingMonitor to model.fit and then call
plot_history to see the loss, its breakdown, your metrics and the sparse diagnostics in one figure.
print_quantization and plot_quantization summarize the HGQ bit-widths.

set_sparse_ebops_factor corrects each sparse conv's EBOPS for the fact that it only computes on the
active pixels, so the monitored EBOPS and the EBOPS regularizer reflect the real sparse cost.
"""

import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import ops
from keras.callbacks import Callback

from .layers import InputReduce, QConv2DSparse


def _input_reduce_layers(model):
    return [layer for layer in model.layers if isinstance(layer, InputReduce)]


def _sparse_conv_scale(model):
    """Map each sparse conv (by its inner QConv2D id) to n_max_pixels / (H*W); {} if not a sparse model."""
    irs = _input_reduce_layers(model)
    if not irs:
        return {}
    n_sparse = irs[0].n_max_pixels
    scale = {}
    for layer in model.layers:
        if not isinstance(layer, QConv2DSparse):
            continue
        try:
            inp = layer.input
            xin = inp[0] if isinstance(inp, (list, tuple)) else inp
            hw = int(xin.shape[1]) * int(xin.shape[2])
        except Exception:
            continue
        if hw > 0:
            scale[id(layer.conv)] = min(1.0, n_sparse / hw)
    return scale


def set_sparse_ebops_factor(model):
    """Correct each sparse conv's EBOPS for the fact that it only computes on the active pixels.

    HGQ counts a conv's EBOPS as if it ran over the whole H*W feature map, which overestimates a
    sparse conv that only touches about n_max_pixels of them. This scales each sparse conv's EBOPS by
    n_max_pixels / (H*W) through HGQ's own ebops_factor, so both the reported EBOPS and the EBOPS
    regularization loss reflect the sparse compute. SparseTrainingMonitor applies this automatically
    at the start of training, so you usually do not need to call it directly -- call it manually only
    to correct EBOPS you inspect before training, or to refresh it if a learnable budget moves a lot
    (it uses the current n_max_pixels, a good approximation since the budget moves slowly).

    Returns a dict of {layer_name: factor} for the sparse convs that were adjusted.
    """
    scale = _sparse_conv_scale(model)
    applied = {}
    for layer in model.layers:
        if isinstance(layer, QConv2DSparse) and id(layer.conv) in scale:
            layer.conv.ebops_factor = scale[id(layer.conv)]
            applied[layer.name] = scale[id(layer.conv)]
    return applied


def _total_ebops(model):
    """Sum HGQ's per-layer EBOPS over the model; returns (total, number of layers with EBOPS)."""
    total, found = 0.0, 0
    try:
        layers = list(model._flatten_layers())
    except Exception:
        layers = list(model.layers)
    for layer in layers:
        e = getattr(layer, "ebops", None)
        if e is None:
            continue
        try:
            val = float(ops.convert_to_numpy(e))
        except Exception:
            try:
                val = float(e)
            except Exception:
                continue
        found += 1
        total += val
    return total, found


def _ebops_reg_loss(model):
    """Sum ebops * beta over the model's HGQ layers, i.e. the EBOPS regularization loss."""
    total = 0.0
    try:
        layers = list(model._flatten_layers())
    except Exception:
        layers = list(model.layers)
    for layer in layers:
        e = getattr(layer, "ebops", None)
        b = getattr(layer, "beta", None)
        if e is None or b is None:
            continue
        try:
            total += float(ops.convert_to_numpy(e)) * float(ops.convert_to_numpy(b))
        except Exception:
            continue
    return total


def _budget_reg_loss(model):
    """Sum beta_n * n over the learnable InputReduce layers, i.e. the pixel-budget penalty."""
    total = 0.0
    for ir in _input_reduce_layers(model):
        if getattr(ir, "learn_n", False):
            total += ir.beta_n * ir.n_max_pixels
    return total


def _masked_intensity_reg_loss(model):
    """Sum beta_maskedE * masked_intensity over the InputReduce layers using the masked-intensity penalty."""
    total = 0.0
    for ir in _input_reduce_layers(model):
        beta_maskedE = getattr(ir, "beta_maskedE", 0.0)
        mi = getattr(ir, "_masked_intensity", None)
        if beta_maskedE > 0 and mi is not None:
            try:
                total += beta_maskedE * float(ops.convert_to_numpy(mi))
            except Exception:
                continue
    return total


def _fmt_bits(bits_tensor):
    b = np.array(bits_tensor).flatten()
    if b.size == 1:
        v = f"{b[0]:.1f}"
        return v, v, v
    return f"{np.mean(b):.1f}", f"{np.min(b):.1f}", f"{np.max(b):.1f}"


def _get_layer_info(layer):
    """Pull the HGQ quantizers, bit-widths and EBOPS from a layer, or None if it has none."""
    if hasattr(layer, "conv") and hasattr(layer.conv, "_kq"):  # QConv2DSparse wraps the conv
        core = layer.conv
        kernel = core.kernel
        bias = getattr(layer, "sparse_bias", None)
        bq = getattr(layer, "_bq", None)
    elif hasattr(layer, "_kq"):  # plain QConv2D / QDense
        core = layer
        kernel = core.kernel
        bias = core.bias if core.use_bias else None
        bq = getattr(core, "_bq", None)
    else:
        return None
    ebops = float(core._ebops) if getattr(core, "_ebops", None) is not None else None
    return dict(
        name=layer.name,
        n_kernel=int(np.prod(kernel.shape)),
        n_bias=int(np.prod(bias.shape)) if bias is not None else 0,
        kq=getattr(core, "_kq", None),
        bq=bq,
        iq=getattr(core, "_iq", None),
        ebops=ebops,
    )


class SparseTrainingMonitor(Callback):
    """Record sparse-model diagnostics into the training history each epoch.

    Add it to the callbacks in model.fit; it stores these in the History so plot_history can show
    them next to the loss and your compiled metrics:

    - n_max_pixels and threshold: the learned or fixed pixel budget and threshold of the InputReduce
      layer (suffixed with the layer name when there is more than one).
    - ebops: total EBOPS over the quantized layers, a proxy for the quantized hardware cost. This
      monitor applies set_sparse_ebops_factor at the start of training, so the EBOPS (and its
      regularizer) reflect the sparse (active-pixel) compute without any manual call.
    - loss_task, loss_ebops, loss_n, loss_maskedE: the training loss split into the task loss, the
      EBOPS penalty, the pixel-budget penalty and the masked-intensity penalty (the last present only
      when its weight beta_maskedE is nonzero), each already scaled by its weight so the parts add up to the
      loss.
    """

    def on_train_begin(self, logs=None):
        # Fold in set_sparse_ebops_factor so the user need not call it manually. Uses the current
        # (initial) budget, matching a manual call right before fit; call it again yourself if a
        # learnable budget moves a lot during training.
        set_sparse_ebops_factor(self.model)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        irs = _input_reduce_layers(self.model)
        multi = len(irs) > 1
        for ir in irs:
            suffix = f"_{ir.name}" if multi else ""
            logs[f"n_max_pixels{suffix}"] = ir.n_max_pixels
            logs[f"threshold{suffix}"] = ir.threshold

        ebops, found = _total_ebops(self.model)
        reg_ebops = _ebops_reg_loss(self.model) if found else 0.0
        has_budget = any(getattr(ir, "learn_n", False) for ir in irs)
        reg_n = _budget_reg_loss(self.model) if has_budget else 0.0
        has_maskedE = any(getattr(ir, "beta_maskedE", 0.0) > 0 for ir in irs)
        reg_maskedE = _masked_intensity_reg_loss(self.model) if has_maskedE else 0.0
        if found:
            logs["ebops"] = ebops
            logs["loss_ebops"] = reg_ebops
        if has_budget:
            logs["loss_n"] = reg_n
        if has_maskedE:
            logs["loss_maskedE"] = reg_maskedE
        # Task loss is the total minus the (train-only) penalties, so the parts add up to the loss.
        if "loss" in logs:
            logs["loss_task"] = logs["loss"] - reg_ebops - reg_n - reg_maskedE


def plot_history(history, early_stopping=None, figsize=None, ncols=3, save_path=None):
    """Plot the training history recorded by SparseTrainingMonitor in one figure.

    Shows the total loss and each compiled metric (train, and validation when present), the loss
    breakdown (task, EBOPS and budget penalties, in red) and the sparse diagnostics (n_max_pixels,
    threshold, ebops, in green). Panels are only created for keys that are present, so it works with
    or without a validation set. Single-line panels annotate their first and last values, and if
    early_stopping restored the best weights, a dashed line marks that epoch and its value.

    Args:
        history: the History returned by model.fit, or its .history dict.
        early_stopping: optional EarlyStopping callback; if it restored best weights, that epoch is marked.
        figsize: optional (width, height); chosen from the number of panels if omitted.
        ncols: number of columns in the panel grid.
    """
    h = history.history if hasattr(history, "history") else dict(history)

    restored = None
    if early_stopping is not None and getattr(early_stopping, "restore_best_weights", False):
        best = getattr(early_stopping, "best_epoch", None)
        if best is not None and best >= 0:
            restored = best + 1

    def _is_single(k):
        return (
            k in ("ebops", "loss_task", "loss_ebops", "loss_n", "loss_maskedE")
            or k.startswith("n_max_pixels")
            or k.startswith("threshold")
        )

    single_keys = [k for k in h if _is_single(k)]
    skip = set(single_keys) | {"lr", "learning_rate"}
    metric_keys = [k for k in h if not k.startswith("val_") and k not in skip]
    m_order = ["loss"]
    metric_keys = [k for k in m_order if k in metric_keys] + [k for k in metric_keys if k not in m_order]
    s_order = ["loss_task", "loss_ebops", "loss_n", "loss_maskedE", "n_max_pixels", "threshold", "ebops"]
    single_keys = [k for k in s_order if k in single_keys] + [k for k in single_keys if k not in s_order]

    panels = metric_keys + single_keys
    if not panels:
        raise ValueError("history has no recorded quantities to plot")

    n = len(panels)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5 * ncols, 3.5 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, key in zip(axes, panels):
        ep = range(1, len(h[key]) + 1)
        show_legend = False
        if key in metric_keys:
            ax.plot(ep, h[key], label=f"train {key}")
            if f"val_{key}" in h:
                ax.plot(ep, h[f"val_{key}"], label=f"val {key}")
            ax.grid(alpha=0.25)
            show_legend = True
        else:
            vals = list(h[key])
            color = "tab:red" if key in ("loss_task", "loss_ebops", "loss_n", "loss_maskedE") else "tab:green"
            ax.plot(ep, vals, marker=".", color=color)
            # annotate first and last epoch (above the point) so the endpoints are easy to read
            for xi, yi in ((1, vals[0]), (len(vals), vals[-1])):
                ax.annotate(
                    f"{yi:g}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=color
                )
            # annotate the restored-epoch value below the point
            if restored is not None and 1 <= restored <= len(vals):
                ax.annotate(
                    f"{vals[restored - 1]:g}",
                    (restored, vals[restored - 1]),
                    textcoords="offset points",
                    xytext=(0, -13),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="0.35",
                )
            if key == "ebops" and min(vals) > 0:
                ax.set_yscale("log")
            ax.margins(y=0.15)
            ax.grid(True, alpha=0.4)
        if restored is not None:
            ax.axvline(restored, color="0.35", ls="--", lw=1.2, label=f"restored @ epoch {restored}")
            show_legend = True
        if show_legend:
            ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(key)

    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def active_pixels_vs_threshold(
    x, thresholds=None, channel=0, percentiles=(25, 50, 75), plot=True, ax=None, save_path=None
):
    """Study how the number of active pixels per image changes with the threshold.

    For each candidate threshold, counts the active pixels (x[..., channel] > threshold) in every
    image and reports the mean, min, max and the given percentiles across the dataset. Use it to pick
    a threshold and an initial budget n: at your chosen threshold, a high percentile or the max tells
    you how many pixels the busiest images have.

    Args:
        x: image array of shape (N, H, W, C).
        thresholds: thresholds to scan; defaults to 50 points across the data range.
        channel: channel used to decide activity (default 0, matching InputReduce).
        percentiles: percentiles of the per-image active-pixel count to report.
        plot: draw the curves (otherwise only the stats are computed).
        ax: optional Axes to draw into; when given, the figure is not shown here.

    Returns a dict with a threshold array plus mean, min, max and p<k> arrays.
    """
    x = np.asarray(x)
    flat = x[..., channel].reshape(x.shape[0], -1)
    if thresholds is None:
        thresholds = np.linspace(float(flat.min()), float(flat.max()), 50)
    thresholds = np.asarray(thresholds, dtype=float)

    stats = {"threshold": thresholds}
    for key in ("mean", "min", "max", *[f"p{p}" for p in percentiles]):
        stats[key] = np.empty_like(thresholds)
    for i, thr in enumerate(thresholds):
        counts = (flat > thr).sum(axis=1)
        stats["mean"][i] = counts.mean()
        stats["min"][i] = counts.min()
        stats["max"][i] = counts.max()
        for p in percentiles:
            stats[f"p{p}"][i] = np.percentile(counts, p)

    if plot:
        owns_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(thresholds, stats["mean"], color="black", lw=2, label="mean")
        ax.plot(thresholds, stats["min"], "--", alpha=0.7, label="min")
        for p in percentiles:
            ax.plot(thresholds, stats[f"p{p}"], "--", alpha=0.7, label=f"{p}th")
        ax.plot(thresholds, stats["max"], "--", alpha=0.7, label="max")
        ax.set_xlabel("threshold")
        ax.set_ylabel("active pixels per image")
        ax.set_title("active pixels vs threshold")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        if owns_fig:
            if save_path is not None:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.show()
    return stats


def plot_reduced_examples(
    x, n, threshold, indices=None, n_examples=5, channel=0, figsize=None, aspect=None, cmap="viridis", save_path=None
):
    """Show what InputReduce keeps for a given budget n and threshold on a few images.

    Each row shows the original image and, at the same intensity scale, the pixels that are kept (the
    first n active pixels in raster order), annotated with the kept and active counts. Use it to check
    that a candidate n and threshold keep the informative pixels.

    Args:
        x: image array of shape (N, H, W, C).
        n: pixel budget (the first n active pixels are kept).
        threshold: activity threshold on the given channel.
        indices: which images to show; defaults to the first n_examples.
        n_examples: number of images to show when indices is not given.
        channel: channel used to decide activity (default 0).
        figsize: optional (width, height).
        aspect: imshow aspect: None keeps the natural pixel ratio, "auto" stretches each image to
            fill its panel (useful for very wide images), or a float scales the pixel height.
        cmap: colormap for the active pixels; empty pixels (at or below 0) are drawn white. For
            binary images a reversed map like "viridis_r" makes the hits dark on white.
    """
    x = np.asarray(x)
    if indices is None:
        indices = np.arange(min(n_examples, len(x)))
    indices = list(indices)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    fig, axes = plt.subplots(len(indices), 2, figsize=figsize or (6, 3 * len(indices)), squeeze=False)
    for row, idx in enumerate(indices):
        s = x[idx][..., channel]
        active = s > threshold
        rank = np.cumsum(active.reshape(-1))
        keep = (active.reshape(-1) & (rank <= n)).reshape(s.shape)
        n_active, n_kept = int(active.sum()), int(keep.sum())
        vmax = float(s.max()) or 1.0

        panels = [
            (s, f"image #{idx}"),
            (np.where(keep, s, 0.0), f"kept (n={n}, thr={threshold:g}): {n_kept} of {n_active} active"),
        ]
        for a, (img, title) in zip(axes[row], panels):
            a.imshow(
                np.ma.masked_less_equal(img, 0.0),
                cmap=cmap_obj,
                vmin=0,
                vmax=vmax,
                aspect=aspect if aspect is not None else "equal",
                interpolation="nearest",
            )
            a.set_title(title)
            a.set_xticks([])
            a.set_yticks([])
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def freeze_quantization(model, freeze_budget=True):
    """Freeze the quantization (and optionally the pixel budget) at the current learned values.

    Sets every quantizer sublayer non-trainable and zeroes the WRAP integer-part decay, so the
    bit-width configuration stays exactly fixed afterwards; optionally also freezes the InputReduce
    budget. Use for two-stage training: after the quantization-aware stage converges, freeze,
    recompile with a lower learning rate, and fine-tune. The weights then adapt to the final bit
    assignment instead of a moving one, and early stopping effectively monitors the pure task loss
    since the frozen regularizer terms are constants.

    Args:
        model: a keras model built with the quantized (HGQ) layers.
        freeze_budget: also set InputReduce layers non-trainable (a learnable budget n stops moving).

    Returns:
        int: the number of layers set non-trainable. Remember to call model.compile again.
    """
    n_frozen = 0
    for sub in model._flatten_layers(include_self=False):
        cls = sub.__class__
        is_quantizer = "quantizer" in cls.__module__.lower() or "Quantizer" in cls.__name__
        if is_quantizer or (freeze_budget and isinstance(sub, InputReduce)):
            sub.trainable = False
            n_frozen += 1
    for w in model.weights:
        if "i_decay" in w.path:
            w.assign(np.zeros(w.shape))
    return n_frozen


def plot_layer_outputs(model, x, index=0, layers=None, cmap="viridis", aspect="auto", max_channels=16, save_path=None):
    """Plot every layer's output for one input image, channel by channel.

    Feeds one image through the model and draws one figure per layer, with that layer's channels
    stacked as rows on a common intensity scale (empty pixels in white) and the active-pixel count
    annotated per channel. Use it to spot where the sparse signals get over-compressed, e.g. a
    pooling layer merging the few active pixels of a track into far fewer, and adjust the
    architecture accordingly.

    Args:
        model: a functional keras model.
        x: image array of shape (N, H, W, C) or a single image (H, W, C).
        index: which image of x to feed.
        layers: layer names to show; defaults to every layer with a plottable output.
        cmap: colormap for the active values; exact zeros are drawn white.
        aspect: imshow aspect for the feature-map panels ("auto" fills each panel, None keeps the
            natural pixel ratio, a float scales the pixel height).
        max_channels: cap on the number of channels drawn per layer.
        save_path: optional path; each layer's figure is saved with the layer name appended.
    """
    x = np.asarray(x, dtype="float32")
    if x.ndim == 3:
        x = x[None]
    xb = x[index : index + 1]

    outputs = {}
    for layer in model.layers:
        if layer.__class__.__name__ == "InputLayer":
            continue
        if layers is not None and layer.name not in layers:
            continue
        try:
            out = layer.output
        except Exception:
            continue
        if isinstance(out, (list, tuple)):
            out = out[0]
        outputs[layer.name] = out
    probe = keras.Model(model.inputs, outputs)
    acts = probe.predict(xb, verbose=0)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    for name, act in acts.items():
        a = np.asarray(act)[0]
        if a.ndim == 1:
            a = a[None, :, None]
        elif a.ndim != 3:
            continue
        n_ch = min(a.shape[-1], max_channels)
        vmin, vmax = min(0.0, float(a.min())), float(a.max()) or 1.0

        fig, axes = plt.subplots(n_ch, 1, figsize=(14, 0.75 * n_ch + 0.6), squeeze=False)
        for ch in range(n_ch):
            img, ax = a[..., ch], axes[ch][0]
            ax.imshow(
                np.ma.masked_equal(img, 0.0),
                cmap=cmap_obj,
                vmin=vmin,
                vmax=vmax,
                aspect=aspect if aspect is not None else "equal",
                interpolation="nearest",
                origin="lower",
            )
            ax.set_ylabel(f"ch {ch}\n{int((img != 0).sum())} act", fontsize=7, rotation=0, ha="right", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        shown = f", showing {n_ch} of {a.shape[-1]} channels" if n_ch < a.shape[-1] else ""
        axes[0][0].set_title(f"{name}: output {tuple(np.asarray(act).shape[1:])}{shown}", loc="left", fontsize=10)
        fig.tight_layout()
        if save_path is not None:
            root, ext = os.path.splitext(save_path)
            fig.savefig(f"{root}_{name}{ext or '.png'}", dpi=150, bbox_inches="tight")
        plt.show()


def print_quantization(model):
    """Print the per-layer quantizer bit-widths (kernel, bias, input as mean, min, max) and EBOPS.

    The EBOPS column reads HGQ's per-layer value, which reflects each layer's ebops_factor, so it is
    sparse corrected once set_sparse_ebops_factor has been applied.
    """
    print(f"\nModel: {model.name}")
    header = (
        f"{'Layer':<12} {'#Kernel':>8} {'#Bias':>6}"
        f"  {'K mean':>6} {'min':>5} {'max':>5}"
        f"  {'B mean':>6} {'min':>5} {'max':>5}"
        f"  {'I mean':>6} {'min':>5} {'max':>5}"
        f"  {'eBOPs':>10}"
    )
    print(header)
    print("-" * len(header))
    total_ebops = 0.0
    for layer in model.layers:
        info = _get_layer_info(layer)
        if info is None:
            continue
        km, klo, khi = _fmt_bits(info["kq"].bits) if info["kq"] else ("-", "-", "-")
        bm, blo, bhi = _fmt_bits(info["bq"].bits) if info["bq"] else ("-", "-", "-")
        im, ilo, ihi = _fmt_bits(info["iq"].bits) if info["iq"] else ("-", "-", "-")
        eb = info["ebops"]
        if eb is not None:
            total_ebops += eb
            es = f"{eb:.0f}"
        else:
            es = "-"
        print(
            f"{info['name']:<12} {info['n_kernel']:>8} {info['n_bias']:>6}"
            f"  {km:>6} {klo:>5} {khi:>5}"
            f"  {bm:>6} {blo:>5} {bhi:>5}"
            f"  {im:>6} {ilo:>5} {ihi:>5}"
            f"  {es:>10}"
        )
    print("-" * len(header))
    print(f"{'Total eBOPs':>{len(header) - 10}}{total_ebops:>10.0f}")


def plot_quantization(models, figsize=(14, 3), save_path=None):
    """Violin plots of the per-layer kernel, bias and input bit-width distributions.

    Args:
        models: a model or list of models to compare (one row per model).
        figsize: (width, per-model height) of the figure.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    categories = [("kq", "Kernel bits"), ("bq", "Bias bits"), ("iq", "Input bits")]
    fig, axes = plt.subplots(
        len(models),
        len(categories),
        figsize=(figsize[0], figsize[1] * len(models)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, model in enumerate(models):
        infos = [info for layer in model.layers if (info := _get_layer_info(layer)) is not None]
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(infos), 1)))
        for col, (key, title) in enumerate(categories):
            ax = axes[row][col]
            names, data, used_colors = [], [], []
            for info, color in zip(infos, colors):
                if info[key] is None:
                    continue
                names.append(info["name"])
                data.append(np.array(info[key].bits).flatten())
                used_colors.append(color)
            if not data:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", fontsize=14, color="gray")
                ax.set_title(f"{model.name} - {title}")
                continue
            parts = ax.violinplot(data, positions=range(len(data)), vert=False, showmedians=True, showextrema=False)
            for body, c in zip(parts["bodies"], used_colors):
                body.set_facecolor(c)
                body.set_alpha(0.7)
            parts["cmedians"].set_color("black")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel("Bitwidth")
            allb = np.concatenate([d.ravel() for d in data])
            lo, hi = float(allb.min()), float(allb.max())
            pad = max(0.5, 0.08 * (hi - lo))  # show the full band with a small margin at both ends
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_title(f"{model.name} - {title}")
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
