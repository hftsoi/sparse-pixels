#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_sparsepixels.h"

// hls-fpga-machine-learning insert weights
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/w13.h"
#include "weights/b13.h"
#include "weights/w16.h"
#include "weights/b16.h"


// hls-fpga-machine-learning insert layer-config
// input_reduce
struct config2 {
    static const unsigned in_height = 48;
    static const unsigned in_width = 48;
    static const unsigned n_chan = 1;
    static const unsigned n_sparse = 20;
    static const unsigned hash_bits = 6;
};

// conv1
struct config4 {
    static const unsigned n_sparse = 20;
    static const unsigned n_chan = 1;
    static const unsigned n_filt = 2;
    static const unsigned kernel_size = 7;
    typedef conv1_accum_t accum_t;
};

// conv1_relu
struct config5 {
    static const unsigned n_sparse = 20;
    static const unsigned n_chan = 2;
};

// pool1
struct config6 {
    static const unsigned n_sparse = 20;
    static const unsigned n_chan = 2;
    static const unsigned pool_size = 4;
    typedef pool1_accum_t accum_t;
};

// conv2
struct config8 {
    static const unsigned n_sparse = 20;
    static const unsigned n_chan = 2;
    static const unsigned n_filt = 3;
    static const unsigned kernel_size = 5;
    typedef conv2_accum_t accum_t;
};

// conv2_relu
struct config9 {
    static const unsigned n_sparse = 20;
    static const unsigned n_chan = 3;
};

// pool2
struct config10 {
    static const unsigned n_sparse = 20;
    static const unsigned n_chan = 3;
    static const unsigned pool_size = 2;
    typedef pool2_accum_t accum_t;
};

// flatten
struct config11 {
    static const unsigned n_sparse = 20;
    static const unsigned n_chan = 3;
    static const unsigned out_height = 6;
    static const unsigned out_width = 6;
};

// dense1
struct config13 : nnet::dense_config {
    static const unsigned n_in = 108;
    static const unsigned n_out = 36;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 3038;
    static const unsigned n_nonzeros = 850;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef dense1_accum_t accum_t;
    typedef dense1_bias_t bias_t;
    typedef dense1_weight_t weight_t;
    typedef layer13_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense1_relu
struct relu_config14 : nnet::activ_config {
    static const unsigned n_in = 36;
    static const unsigned table_size = 8192;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef dense1_relu_table_t table_t;
};

// dense2
struct config16 : nnet::dense_config {
    static const unsigned n_in = 36;
    static const unsigned n_out = 10;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 96;
    static const unsigned n_nonzeros = 264;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef dense2_accum_t accum_t;
    typedef dense2_bias_t bias_t;
    typedef dense2_weight_t weight_t;
    typedef layer16_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// softmax
struct softmax_config17 : nnet::activ_config {
    static const unsigned n_in = 10;
    static const unsigned n_slice = 10;
    static const unsigned n_outer = 1;
    static const unsigned n_inner = 1;
    static const unsigned parallelization_factor = -1;
    static const unsigned exp_table_size = 1024;
    static const unsigned inv_table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    static constexpr float exp_scale = 1.0;
    typedef softmax_exp_table_t exp_table_t;
    typedef softmax_inv_table_t inv_table_t;
    typedef model_default_t accum_t;
    typedef softmax_inv_inp_t inv_inp_t;
    typedef ap_ufixed<13, 8> inp_norm_t;
};



#endif
