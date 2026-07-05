# Installation

With Python >= 3.10:

``` bash
pip install sparsepixels
```

This pulls in everything you need to build and train sparse models: TensorFlow, Keras 3, HGQ2 (the quantization backend), and our built-in monitoring and plotting tools.

## Converting to HLS

Deploying a trained model to FPGA firmware additionally needs [hls4ml](https://github.com/fastmachinelearning/hls4ml). Support for `sparsepixels` is in an open [pull request](https://github.com/fastmachinelearning/hls4ml/pull/1468) that is not yet merged, so for now install hls4ml from the PR branch:

``` bash
pip install "git+https://github.com/hftsoi/hls4ml.git@sparsepixels"
```

See [HLS Conversion](conversion.md) for details.
