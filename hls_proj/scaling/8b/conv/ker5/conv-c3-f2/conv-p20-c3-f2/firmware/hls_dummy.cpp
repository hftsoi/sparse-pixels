#include <iostream>

#include "hls_dummy.h"
#include "parameters.h"

//template <class T, class t> class Op_active {
//    public:
//      T operator()(T a, T b, t threshold) { return (a.value > threshold) ? a : b; }
//};

template <class T, class t> class Op_active { // test with old Op_active and setting the check in main fill loop
    public:
      T operator()(T a, T b, t threshold) {
        if (a.value > threshold) return a;
        else if (b.value > threshold) return b;
        else {
            T none;
            none.value = 0;
            none.index = 0;
            return none;
        }
      }
};
  
constexpr int _floorlog2(int x) { return (x < 2) ? 0 : 1 + _floorlog2(x / 2); }
constexpr int _pow2(int x) { return x == 0 ? 1 : 2 * _pow2(x - 1); }

template <typename T>
struct value_idx_pair {
    T value;
    ap_uint<12> index;
};

template <class T, int N, class Op, class t> T find_active(T *x, Op op, t threshold) {
#pragma HLS INLINE
    // This function finds the leftmost nonzero element in an array.

    // Split the array with two subarrays.
    // leftN: largest power of 2 that is less than N.
    static constexpr int leftN = _pow2(_floorlog2(N - 1)) > 0 ? _pow2(_floorlog2(N - 1)) : 0;
    static constexpr int rightN = N - leftN > 0 ? N - leftN : 0;

    if (N == 1) {
        return x[0];
    }
    if (N == 2) {
        return op(x[0], x[1], threshold);
    }
    return op(find_active<T, leftN, Op, t>(x, op, threshold), find_active<T, rightN, Op, t>(x + leftN, op, threshold), threshold);
}


template <class data_T, class hash_T, int N_h, int N_w, int N_c, int N_sparse>
void sparse_input_reduce(data_T input_arr[N_h * N_w * N_c],
                         data_T threshold,
                         data_T sparse_arr_feat[N_sparse * N_c],
                         hash_T sparse_arr_hash[N_sparse * 2]) {
    // This function reduces the full input array into sparse arrays, storing active inputs (nonzeros).
    // 1) sparse_arr_feat stores features (N_c channels) from the active pixels.
    // 2) sparse_arr_hash stores the corresponding pixel height and width indices, in the same ordering as sparse_arr_feat.
    // The sparse arrays assume a fixed max number of active pixels considered in a dataset (N_sparse).
    // For simplicity, scan for active pixels in the first channel only.
    // If a pixel is active in the first channel, then store the features from all channels.

    value_idx_pair<data_T> pair_arr[N_h * N_w];
    int j_h_arr[N_h * N_w];
    int j_w_arr[N_h * N_w];
    #pragma HLS ARRAY_PARTITION variable=j_h_arr type=complete dim=0
    #pragma HLS ARRAY_PARTITION variable=j_w_arr type=complete dim=0
    #pragma HLS ARRAY_PARTITION variable=pair_arr type=complete dim=0

    // Loop over pixels and get features from the first channel, also compute height and width indices.
    DataPrepareLoop:
    for (int j = 0; j < N_h * N_w; j++) {
        #pragma HLS UNROLL
        pair_arr[j].value = input_arr[N_c * j];
        pair_arr[j].index = j;

        int pixels_per_channel = N_h * N_w;
        //int j_c = j / pixels_per_channel + 1;
        int remainder = j % pixels_per_channel;
        int j_h = remainder / N_h + 1;
        int j_w = remainder % N_w + 1;

        j_h_arr[j] = j_h;
        j_w_arr[j] = j_w;
        // ^ I tried pre-computing these and storing in row_arr[N_h * N_w] and col_arr[N_h * N_w]
        // but resources stayed exactly the same
    }

    // Find active pixels and fill the sparse arrays.
    Op_active<value_idx_pair<data_T>, data_T> op_active;
    MaxPixelsLoop:
    for (int i = 0; i < N_sparse; i++) {
        #pragma HLS PIPELINE
        // Iteratively find the leftmost active element in the input array.
        value_idx_pair<data_T> pair = find_active<value_idx_pair<data_T>,
                                                  N_h * N_w,
                                                  Op_active<value_idx_pair<data_T>, data_T>,
                                                  data_T>(pair_arr, op_active, threshold);
        // Fill feature for the first channel.
        sparse_arr_feat[N_c * i] = pair.value;
        // Other channels, if there is any.
        for (int j = 1; j < N_c; j++) {
            #pragma HLS UNROLL
            sparse_arr_feat[N_c * i + j] = input_arr[N_c * pair.index + j];
        }
        // ^ probably set >threshold check here than modifying Op_active, since usually N_c=1 is cheap?

        // Fill height and width indices.
        sparse_arr_hash[2 * i] = j_h_arr[pair.index]; 
        sparse_arr_hash[2 * i + 1] = j_w_arr[pair.index];

        // Update the filled active feature to zero so the next find_active can find the next active.
        pair_arr[pair.index].value = 0;
    }
    // Note that sparse_arr_hash[2 * i] and sparse_arr_hash[2 * i + 1] can fill arbitrary numbers even when the corresponding
    // sparse_arr_feat[N_c * i], sparse_arr_feat[N_c * i + 1], ..., sparse_arr_feat[N_c * i + N_c - 1] are zeros (inactive).
    // This will be the case when the number of active pixels is less then N_sparse in an example.
    // These "zero-padded" pixels will be ignored as the following sparse operations will only be effective on nonzero elements in sparse_arr_feat.
}

template <class data_T, class res_T, class w_T, int n_chan, int n_filt, int N_sparse, int ker_size>
res_T mult_for_sparse_conv_kernel(int offset_h, int offset_w, data_T sparse_arr_feat_in[n_chan * N_sparse], w_T filt_w[ker_size * ker_size * n_chan * n_filt], int i_filt, int i_pixel_in) {
    // Helper function for sparse_conv, for a kernel size of 3.
    // It picks the correct filter weights when multiplying the input features over channels, given a filter position and a pixel position.
    // Note the ordering in weight array: [p1-f1-c1, p1-f1-c2, p1-f2-c1, p1-f2-c2, p2-f1-c1,...],
    // if there are two input channels and two output filters.
    // Here w_idx computes the relative weight index for a given filter and channel, disregarding the pixel offset position.
    #pragma HLS INLINE

    // Valid offset positions.
    constexpr int R = (ker_size - 1) / 2;
    if ( (unsigned)(offset_h + R) >= ker_size || (unsigned)(offset_w + R) >= ker_size ) { return (res_T)0; }
    ap_uint<4> row = R - offset_h;
    ap_uint<4> col = R - offset_w;
    // Unroll 2d filter location into flat index like 0..8 for ker_size=3 (from top left to right and top to down, so center is 4)
    ap_uint<7> pos = row * ker_size + col;

    // Weights per spatial position
    const int w_idx = n_chan * i_filt + n_filt * n_chan * pos;

    // Accumulate over channels
    res_T acc = 0;
    MultLoopPerFilter:
    for (int i_chan = 0; i_chan < n_chan; i_chan++) {
        #pragma HLS UNROLL
        acc += filt_w[w_idx + i_chan] * sparse_arr_feat_in[n_chan * i_pixel_in + i_chan];
    }
    return acc;
}

template <class data_T, class res_T, class hash_T, class w_T, class b_T, int N_sparse, int n_chan, int n_filt, int ker_size>
void sparse_conv(data_T sparse_arr_feat_in[N_sparse * n_chan],
                 res_T sparse_arr_feat_out[N_sparse * n_filt],
                 hash_T sparse_arr_hash[N_sparse * 2],
                 w_T w[ker_size * ker_size * n_chan * n_filt],
                 b_T b[n_filt]) {
    // Note the ordering of filter weights stored by hls4ml,
    // pixel_loop->filter_loop->channel_loop, so shape=(h, w, f, c) fattened.
    // If there are two input channels and two output filters:
    // [p1-f1-c1, p1-f1-c2, p1-f2-c1, p1-f2-c2, p2-f1-c1,...].

    // For sparse convolution, we assume the same sparsity structure between input and output,
    // which reduces computing complexity and avoid the dilation problem.
    // So the sparse_arr_hash stays exactly the same, we just compute the new feature values for the same set of pixel locations.

    // Loop over output pixels (same set of input pixels).
    OutputPixelLoop:
    for (int i_pixel_out = 0; i_pixel_out < N_sparse; i_pixel_out++) {
        #pragma HLS UNROLL

        bool nonzero = false;
        for (int i_chan = 0; i_chan < n_chan; i_chan++) {
            #pragma HLS UNROLL
            nonzero |= (sparse_arr_feat_in[i_pixel_out * n_chan + i_chan] != (data_T)0);
        }

        // Loop over output filters.
        OutputFilterLoop:
        for (int i_filt = 0; i_filt < n_filt; i_filt++) {
            #pragma HLS UNROLL
            res_T acc = 0;
        
            // Loop over input pixels for the current output pixel.
            InputPixelLoop:
            for (int i_pixel_in = 0; i_pixel_in < N_sparse; i_pixel_in++) {
                #pragma HLS UNROLL

                // Compute the relative positions between the output and input pixels in height and width.
                int offset_h = sparse_arr_hash[2 * i_pixel_out] - sparse_arr_hash[2 * i_pixel_in];
                int offset_w = sparse_arr_hash[2 * i_pixel_out + 1] - sparse_arr_hash[2 * i_pixel_in + 1];

                acc += mult_for_sparse_conv_kernel<data_T, res_T, w_T, n_chan, n_filt, N_sparse, ker_size>(offset_h, offset_w, sparse_arr_feat_in, w, i_filt, i_pixel_in);
            }

            // Add bias.
            if (acc != 0) { acc += b[i_filt]; }
            if (nonzero == false) { acc = 0; }
            sparse_arr_feat_out[n_filt * i_pixel_out + i_filt] = acc;
        }
    }
}

template <class data_T, class res_T, int N_sparse, int n_chan>
void sparse_relu(data_T sparse_arr_feat_in[N_sparse * n_chan], res_T sparse_arr_feat_out[N_sparse * n_chan]) {
    #pragma HLS PIPELINE
    data_T data;
    for (int i = 0; i < N_sparse * n_chan; i++) {
        data = sparse_arr_feat_in[i];
        if (data > 0) {
            sparse_arr_feat_out[i] = data;
        } else {
            sparse_arr_feat_out[i] = 0;
        }
    }
}

template <class data_T, class res_T, class hash_T, int N_sparse, int n_chan, int pool_size>
void sparse_pooling_avg(data_T sparse_arr_feat_in[N_sparse * n_chan],
                        res_T sparse_arr_feat_out[N_sparse * n_chan],
                        hash_T sparse_arr_hash_in[N_sparse * 2],
                        hash_T sparse_arr_hash_out[N_sparse * 2]) {
    // This function does average pooling on sparse arrays.
    // It first computes the pooled positions for all active pixels stored in the sparse arrays,
    // then collects pixels falling in the same pool and computes the average per pool.
    // This therefore gives new sparse_arr_hash and sparse_arr_feat at outputs.

    // Store the 1/pool_size value instead of doing division at runtime.
    ap_fixed<10,0> pool_size_recip = 0;
    if (pool_size == 2) { pool_size_recip = 0.5; }
    else if (pool_size == 3) { pool_size_recip = 0.33333; }
    else if (pool_size == 4) { pool_size_recip = 0.25; }
    else if (pool_size == 5) { pool_size_recip = 0.2; }
    else if (pool_size == 6) { pool_size_recip = 0.16667; }

    // First compute the pooled postitions for all pixels.
    // This is needed here separately first since we need to do nested loops over it later.
    int hash_tmp[N_sparse * 2];
    #pragma HLS ARRAY_PARTITION variable=hash_tmp type=complete dim=0
    ComputePooledLoc:
    for (int i = 0; i < N_sparse; i++) {
        #pragma HLS UNROLL
        //hash_tmp[2 * i] = (sparse_arr_hash_in[2 * i] - 1) * pool_size_recip + 1; // potential casting problem with "* pool_size_recip"
        //hash_tmp[2 * i + 1] = (sparse_arr_hash_in[2 * i + 1] - 1) * pool_size_recip + 1;
        hash_tmp[2 * i] = (sparse_arr_hash_in[2 * i] - 1) / pool_size + 1; // fast when pool_size is 2,4,...
        hash_tmp[2 * i + 1] = (sparse_arr_hash_in[2 * i + 1] - 1) / pool_size + 1;
    }

    // Get a copy of the sparse feature array,
    // as we will need to use it to zero out pixels later for keeping track of when the pooling is done on a pixel to avoid double counting.
    data_T sparse_arr_feat_in_copy[N_sparse * n_chan];
    #pragma HLS ARRAY_PARTITION variable=sparse_arr_feat_in_copy type=complete dim=0
    for (int i = 0; i < N_sparse * n_chan; i++) {
        #pragma HLS UNROLL
        sparse_arr_feat_in_copy[i] = sparse_arr_feat_in[i];
    }

    // Now, the pooling operation.
    // Loop over the pixels.
    HashOutLoop:
    for (int i_pixel = 0; i_pixel < N_sparse; i_pixel++) {
        #pragma HLS UNROLL
        // Get the pooled locations for a given output hash.
        int h_out = hash_tmp[2 * i_pixel];
        int w_out = hash_tmp[2 * i_pixel + 1];

        // Loop over the input channels.
        ChannelLoop:
        for (int i_chan = 0; i_chan < n_chan; i_chan++) {
            #pragma HLS UNROLL
            res_T acc = 0;

            // Loop over given the pixels to find if any belongs to the same pool for a given i_pixel.
            HashInLoop:
            for (int j_pixel = 0; j_pixel < N_sparse; j_pixel++) {
                #pragma HLS UNROLL
                // Get the pooled locations for a given input hash.
                int h_in = hash_tmp[2 * j_pixel];
                int w_in = hash_tmp[2 * j_pixel + 1];

                // Accumulate if the j_pixel belong to the same pool as i_pixel.
                data_T data = sparse_arr_feat_in_copy[n_chan * j_pixel + i_chan];
                if ((h_out == h_in) && (w_out == w_in)) {
                    acc += data;
                    // Zero it out so the next iteration will not double count it.
                    sparse_arr_feat_in_copy[n_chan * j_pixel + i_chan] = 0;
                }
            }
            // Divide the sum by the number pixels within the pool.
            sparse_arr_feat_out[n_chan * i_pixel + i_chan] = acc * pool_size_recip * pool_size_recip;
        }
        // Fill the pixel locations after pooling.
        sparse_arr_hash_out[2 * i_pixel] = h_out;
        sparse_arr_hash_out[2 * i_pixel + 1] = w_out;
    }
    // Note that there could be more zero-padded elements in sparse_arr_feat_out than sparse_arr_feat_in due to pooling,
    // and the new zero-padded elements will have redundant values in hash.
    // But whenever the element is zero-padded in the sparse feat array, it will be ignored by sparse operations no matter what its hash is.
}

template<class data_T, class hash_T, int n_height, int n_width, int n_chan, int N_sparse>
void sparse_flatten(data_T sparse_arr_feat[N_sparse * n_chan],
                    hash_T sparse_arr_hash[N_sparse * 2],
                    data_T flat_arr[n_height * n_width * n_chan]) {
    // This function does full flattening on sparse arrays.
    
    InitFlatArr:
    for (int i = 0; i < n_height * n_width * n_chan; i++) {
        #pragma HLS UNROLL
        flat_arr[i] = 0;
    }

    FillFlatArr:
    for (int i = 0; i < N_sparse; i++) {
        #pragma HLS UNROLL
        // Compute the pixel number from its height and width indices.
        int i_h = sparse_arr_hash[2 * i];
        int i_w = sparse_arr_hash[2 * i + 1];
        int pixel_idx = (i_h - 1) * n_width + (i_w - 1);

        ChannelLoop:
        for (int i_chan = 0; i_chan < n_chan; i_chan++) {
            #pragma HLS UNROLL
            data_T data = sparse_arr_feat[n_chan * i + i_chan];

            // This check is needed as the hash array after pooling could have duplicate pixels with 0 feat val,
            // so it avoids over-writing the duplicate pixels with zeros.
            if (data != 0) {
                flat_arr[n_chan * pixel_idx + i_chan] = data;
            }
        }
    }
}

void hls_dummy(
    input_t x_in[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    result_t layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x_in complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x_in,layer2_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<weight2_t, 27>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 3>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight4_t, 150>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 2>(b4, "b4.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    input_t active_threshold = 1.5;
    input_t sparse_arr_feat_reduce_out[N_MAX_PIXELS];
    ap_uint<10> sparse_arr_hash_reduce_out[N_MAX_PIXELS * 2];
    #pragma HLS ARRAY_PARTITION variable=sparse_arr_feat_reduce_out complete dim=0
    #pragma HLS ARRAY_PARTITION variable=sparse_arr_hash_reduce_out complete dim=0
    sparse_input_reduce<input_t, ap_uint<10>, N_INPUT_1_1, N_INPUT_2_1, N_INPUT_3_1, N_MAX_PIXELS>(x_in, active_threshold, sparse_arr_feat_reduce_out, sparse_arr_hash_reduce_out); // sparse array creation

    result_t sparse_arr_feat_conv1_out[N_MAX_PIXELS * 3];
    #pragma HLS ARRAY_PARTITION variable=sparse_arr_feat_conv1_out complete dim=0
    sparse_conv<input_t, result_t, ap_uint<10>, weight2_t, bias2_t, N_MAX_PIXELS, 1, 3, 3>(sparse_arr_feat_reduce_out, sparse_arr_feat_conv1_out, sparse_arr_hash_reduce_out, w2, b2); // sparse conv1
    
    result_t sparse_arr_feat_conv2_out[N_MAX_PIXELS * 2];
    #pragma HLS ARRAY_PARTITION variable=sparse_arr_feat_conv2_out complete dim=0
    sparse_conv<result_t, result_t, ap_uint<10>, weight4_t, bias4_t, N_MAX_PIXELS, 3, 2, 5>(sparse_arr_feat_conv1_out, sparse_arr_feat_conv2_out, sparse_arr_hash_reduce_out, w4, b4); // sparse conv2

    for (int i = 0; i < N_MAX_PIXELS * 2; i++) {
        #pragma HLS UNROLL
        layer2_out[i] = sparse_arr_feat_conv2_out[i];
    }
}

