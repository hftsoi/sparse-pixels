# API Reference

Auto-generated from the docstrings. See [Quick Start](quick_start.md) for how the pieces fit together.

## Layers

The sparse layers live in `sparsepixels.layers`. `InputReduce` produces the sparse representation, and the rest consume it.

::: sparsepixels.layers.InputReduce
    options:
      members:
        - n_max_pixels
        - threshold

::: sparsepixels.layers.QConv2DSparse
    options:
      members: false

::: sparsepixels.layers.AveragePooling2DSparse
    options:
      members: false

::: sparsepixels.layers.MaxPooling2DSparse
    options:
      members: false

## Data study

Utilities in `sparsepixels.utils` for picking a threshold and budget before training.

::: sparsepixels.utils.active_pixels_vs_threshold

::: sparsepixels.utils.plot_reduced_examples

## Training & monitoring

::: sparsepixels.utils.SparseTrainingMonitor
    options:
      members: false

::: sparsepixels.utils.plot_history

::: sparsepixels.utils.set_sparse_ebops_factor

## Quantization diagnostics

::: sparsepixels.utils.print_quantization

::: sparsepixels.utils.plot_quantization
