# Overview

A detector image is mostly empty. Only a handful of pixels carry a hit, but a standard CNN still convolves over every pixel, so its cost grows with the full image size. SparsePixels keeps only the pixels that matter and convolves around those, so the cost follows the number of hits instead. That is what makes it realistic to run a CNN for real-time inference on an FPGA.

Our design starts from the hardware. On the FPGA an image is never stored as a dense grid; it is carried as a short list of active pixels, and every layer has a HLS kernel that reads and writes that list. The training library is built to mirror those kernels exactly, so the Keras model you train is not a separate thing that gets approximated in hardware later. It is the hardware behavior expressed in Keras, with quantization-aware training layered on top, and a C-simulation of the firmware matches it down to the fixed-point rounding. The two sections below cover the two halves: how you train a model, and what it becomes on the FPGA.

## The training library

You build an ordinary Keras model out of the sparse layers shipped from the SparsePixels library, integrated with HGQ2 quantization scopes.

`InputReduce` is the entry point. From the raw image it produces a feature map masked down to the kept pixels, together with a keep mask, and everything after it (the sparse convolution and pooling layers) works on that active-pixel set. `InputReduce` keeps the first `n` pixels whose value is above `threshold`, scanning the image in row-major order.

The point of making the pixel budget `n` and the `threshold` trainable is that you no longer have to guess them. Training moves them like any other weight. A pixel budget penalty of weight `beta_n` pushes `n` smaller so the model spends fewer active pixels, and a masked-intensity penalty of weight `beta_maskedE` pushes back whenever the model starts throwing away real signal, whether by raising the threshold or by shrinking the budget. The selection itself is always exact and identical to the hardware, so the layer behaves the same whether you are training or deploying. If you would rather fix either value, set `learn_n=False` or `learn_threshold=False`.

Quantization runs throughout via HGQ2: weights and activations carry learnable bit-widths, and EBOPS, a proxy for the quantized hardware cost, is regularized during training with weight `beta0`. Because a sparse convolution only touches about `n` of the pixels, our `SparseTrainingMonitor` rescales each sparse conv's EBOPS by `n / (H*W)` so the reported cost and the regularizer reflect the real sparse compute. The result is a loss with a few named parts that you can watch and balance during training; see [Training & Monitoring](training.md).

## The HLS design

On the FPGA the image lives in two short, fixed-length arrays that together form the sparse representation:

- a **feature array** holding the channel values of the kept pixels, and
- a **coordinate array** holding the (row, column) of each kept pixel.

Their length is the deployed pixel budget, `n_max_pixels`. Every sparse layer is a kernel that reads and writes this pair of arrays:

- `sparse_input_reduce` scans the incoming image, keeps the first `n_max_pixels` pixels above the threshold, and writes out the feature and coordinate arrays.
- `sparse_conv` convolves without ever building a dense buffer. For each kept output pixel it walks over the kept input pixels, and whenever a pair's coordinate offset falls inside the kernel window it adds that multiply-accumulate. The work scales with the number of kept pixels, not with the image area.
- `sparse_pooling_avg` and `sparse_pooling_max` pool by dividing the coordinates down and merging pixels that land in the same output cell.
- `sparse_flatten` scatters the kept pixels back into a dense vector using their coordinates, which is what lets a normal dense layer follow. In a dense CNN a flatten is just a reshape and costs nothing, so hls4ml does not even emit a kernel for it; here it is a real scatter from the sparse arrays back into a dense buffer, and it happens once, at the last sparse layer.

Because each kernel is a fixed loop over the pixel list, you can choose how much of it runs at once. Those choices are exposed as per-layer knobs at conversion time, and they only change latency and resource usage but not outputs. See [HLS Conversion](conversion.md).

## Putting it together

Train a model with the sparse layers with the SparsePixels library, read the learned pixel budget and threshold off `InputReduce` (`n_max_pixels` and `threshold`), and convert with hls4ml. It picks those values up on its own and emits the kernels above, and because the Keras layers mirror the kernels exactly, the C-simulation of the generated firmware matches your trained model.
