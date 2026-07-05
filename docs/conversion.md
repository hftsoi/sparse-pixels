# HLS Conversion with hls4ml

A trained SparsePixels model converts to FPGA firmware through the hls4ml integration. The converter reads the deployed pixel budget and threshold straight off the model and emits the sparse HLS kernels, with per-layer knobs to trade latency against resources.

!!! note "Installation"

    hls4ml support for `sparsepixels` is in an open [pull request](https://github.com/fastmachinelearning/hls4ml/pull/1468) that is not yet merged. For now, install hls4ml from the PR branch:

    ``` bash
    pip install "git+https://github.com/hftsoi/hls4ml.git@sparsepixels"
    ```

## How the firmware runs

It helps to keep in mind what the firmware actually does, since that is what the knobs control. On the FPGA the image is carried as the two short arrays from the [Overview](overview.md): a feature array of pixel values and a coordinate array of (row, column) positions. Their length is fixed at `n_max_pixels`, the pixel budget your `InputReduce` learned during training, so a smaller learned budget means shorter arrays and less computation in every layer downstream.

Each sparse layer is a loop over that pixel list:

- the input reduction scans the image and fills the two arrays,
- the sparse conv loops over the kept output pixels and, inside that, over its filters,
- sparse pooling loops over the kept pixels and over the channels,
- the flatten loops over output positions and scatters the kept pixels into a dense vector. In a dense CNN a flatten is a free reshape, so hls4ml skips it; here it is a real scatter and gets its own kernel.

The knobs below say how many iterations of each of these loops run at once.

## Convert with defaults

Pull a config from the trained model and convert. With no per-layer settings, every sparse layer is fully parallelized, which is the lowest-latency default:

``` python
import hls4ml

hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
hls_config.setdefault('Model', {})['PipelineStyle'] = 'dataflow'  # "#pragma HLS DATAFLOW"

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=hls_config,
    output_dir='hls_proj/my_sparse_cnn',
    backend='Vitis',
    io_type='io_parallel',
)
hls_model.write()    # emit the HLS project to synthesize
hls_model.compile()  # C-sim build for a quick Keras-vs-HLS check
y_hls = hls_model.predict(x_test)
```

The deployed pixel budget and threshold (`layer.n_max_pixels`, `layer.threshold`) are parsed from the model automatically.

## Controlling parallelization

To trade latency for resources, set the knobs on `hls_config` before converting. Which layers are sparse matters here, so this is the model the example targets:

``` python
x_in    = keras.Input((28, 28, 1), name='x_in')
x, mask = InputReduce(..., name='input_reduce')(x_in)            # sparse
x       = QConv2DSparse(filters=3, ..., name='conv1')([x, mask]) # sparse
x, mask = AveragePooling2DSparse(2, name='pool1')([x, mask])     # sparse
x       = keras.layers.Flatten(name='flatten')(x)                # sparse: scatter to dense
x       = QDense(10, name='dense1', ...)(x)                      # ordinary dense, not sparse
x       = keras.layers.Activation('softmax')(x)
```

So the sparse layers you tune are `input_reduce`, `conv1`, `pool1`, and `flatten`. `dense1` is a normal dense layer and takes the usual hls4ml settings. Each factor unrolls a specific loop:

``` python
n_max_pixels = model.get_layer('input_reduce').n_max_pixels

# input reduce: how the selection is built (see below)
#               a standalone knob, not a parallel factor
hls_config['LayerName']['input_reduce']['Variant'] = 'tree' # or 'stream'

# conv1: PixelParallelFactor unrolls the active-pixel loop
#        FiltParallelFactor the filter loop
hls_config['LayerName']['conv1']['PixelParallelFactor'] = n_max_pixels
hls_config['LayerName']['conv1']['FiltParallelFactor'] = 3

# pool1: PixelParallelFactor unrolls the active-pixel loop
#        ChanParallelFactor the channel loop
hls_config['LayerName']['pool1']['PixelParallelFactor'] = n_max_pixels
hls_config['LayerName']['pool1']['ChanParallelFactor'] = 3

# flatten: ParallelFactor unrolls the scatter loop
#          over output positions (out_height * out_width)
hls_config['LayerName']['flatten']['ParallelFactor'] = 14 * 14

# ...then convert exactly as in the default example above.
```

| Layer | Knob | Loop it unrolls |
|-------|------|-----------------|
| conv | `PixelParallelFactor` | over the kept active pixels (up to `n_max_pixels`) |
| conv | `FiltParallelFactor` | over the output filters |
| pool | `PixelParallelFactor` | over the kept active pixels |
| pool | `ChanParallelFactor` | over the channels |
| flatten | `ParallelFactor` | over the scatter positions (up to `out_height * out_width`) |

Higher factors mean lower latency and more resources.

## Tree vs streaming input reduction

The input reduction has one knob of its own, `Variant`, and it is separate from the parallel factors above. It picks between two ways of building the sparse arrays, at opposite ends of the latency and resource trade:

- `tree` (default) selects the kept pixels with a parallel reduction: lowest latency, most resources.
- `stream` selects them sequentially one at a time: higher latency (at least `H*W` clocks), least resources.

This only changes how the input reduction itself is built. It does not touch any downstream layer, so you can flip it on its own. Unfortunately these two are the only options at the moment, as we have tested some kernels that run in the middle of the two extremes (e.g., chunk scan), but so far not successful without introducing data-dependent writes.

## Choosing a factor

A factor that divides its loop count unrolls cleanly; a non-divisor still works, since the kernel handles the remainder, but is a little less efficient. Because the result is identical for any setting, start from the defaults and dial factors down only where you need to fit the resource budget.

## Verifying the conversion

`hls_model.compile()` builds a C-simulation of the generated firmware, so you can check it against Keras before synthesis:

``` python
y_keras = model.predict(x_test)
y_hls = hls_model.predict(x_test).reshape(y_keras.shape)
print('max abs difference:', np.max(np.abs(y_keras - y_hls)))
```

For a properly trained model the two agree down to the fixed-point rounding of the quantized model, which is the whole point of building the training layers to mirror the kernels.
