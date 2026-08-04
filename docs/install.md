# Installation

With Python >= 3.10:

``` bash
pip install sparsepixels
```

This pulls in everything you need to build and train sparse models: TensorFlow, Keras 3, HGQ2 (the quantization backend), and our built-in monitoring and plotting tools.

## Converting to HLS

Deploying a trained model to FPGA firmware additionally needs [hls4ml](https://github.com/fastmachinelearning/hls4ml). Support for `sparsepixels` is merged into hls4ml ([PR #1468](https://github.com/fastmachinelearning/hls4ml/pull/1468)) but is not in a tagged release yet, so install hls4ml from the main branch:

``` bash
pip install "git+https://github.com/fastmachinelearning/hls4ml.git"
```

Once a release including it is out, plain `pip install hls4ml` will be enough.

See [HLS Conversion](conversion.md) for details.
