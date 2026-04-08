#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    x_in_t x_in[48*48*1],
    result_t layer17_out[10]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x_in complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x_in,layer17_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<conv1_weight_t, 49>(w4, "w4.txt");
        nnet::load_weights_from_txt<conv1_bias_t, 1>(b4, "b4.txt");
        nnet::load_weights_from_txt<conv2_weight_t, 75>(w8, "w8.txt");
        nnet::load_weights_from_txt<conv2_bias_t, 3>(b8, "b8.txt");
        nnet::load_weights_from_txt<dense1_weight_t, 3888>(w13, "w13.txt");
        nnet::load_weights_from_txt<dense1_bias_t, 36>(b13, "b13.txt");
        nnet::load_weights_from_txt<dense2_weight_t, 360>(w16, "w16.txt");
        nnet::load_weights_from_txt<dense2_bias_t, 10>(b16, "b16.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    input_reduce_t layer2_out[20];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    conv1_iq_t layer3_out[20];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    conv1_t layer4_out[20];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    conv1_relu_t layer5_out[20];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    pool1_t layer6_out[20];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    conv2_iq_t layer7_out[20];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    conv2_t layer8_out[60];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    conv2_relu_t layer9_out[60];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    pool2_t layer10_out[60];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    flatten_t layer11_out[108];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0

    dense1_iq_t layer12_out[108];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    dense1_t layer13_out[36];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0

    dense1_relu_t layer14_out[36];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0

    dense2_iq_t layer15_out[36];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0

    dense2_t layer16_out[10];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0

    x_in_t threshold_2 = 0.4;
ap_uint<6> sparse_hash_input_reduce[20 * 2];
#pragma HLS ARRAY_PARTITION variable=sparse_hash_input_reduce complete dim=0
sparse_input_reduce<x_in_t, input_reduce_t, ap_uint<6>, 48, 48, 1, 20>(x_in, threshold_2, layer2_out, sparse_hash_input_reduce); // input_reduce
#ifndef __SYNTHESIS__
    nnet::save_layer_output<input_reduce_t>(layer2_out, "input_reduce", 20);
#endif

    nnet::conv1_iq<input_reduce_t, conv1_iq_t>(layer2_out, layer3_out); // conv1_iq
#ifndef __SYNTHESIS__
    nnet::save_layer_output<conv1_iq_t>(layer3_out, "conv1_iq", 20);
#endif

    sparse_conv<conv1_iq_t, conv1_t, ap_uint<6>, conv1_weight_t, conv1_bias_t, conv1_accum_t, 20, 1, 1, 7>(layer3_out, layer4_out, sparse_hash_input_reduce, w4, b4); // conv1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<conv1_t>(layer4_out, "conv1", 20);
#endif

    sparse_relu<conv1_t, conv1_relu_t, 20, 1>(layer4_out, layer5_out); // conv1_relu
#ifndef __SYNTHESIS__
    nnet::save_layer_output<conv1_relu_t>(layer5_out, "conv1_relu", 20);
#endif

    ap_uint<6> sparse_hash_pool1[20 * 2];
#pragma HLS ARRAY_PARTITION variable=sparse_hash_pool1 complete dim=0
sparse_pooling_avg<conv1_relu_t, pool1_t, ap_uint<6>, pool1_accum_t, 20, 1, 4>(layer5_out, layer6_out, sparse_hash_input_reduce, sparse_hash_pool1); // pool1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<pool1_t>(layer6_out, "pool1", 20);
#endif

    nnet::conv2_iq<pool1_t, conv2_iq_t>(layer6_out, layer7_out); // conv2_iq
#ifndef __SYNTHESIS__
    nnet::save_layer_output<conv2_iq_t>(layer7_out, "conv2_iq", 20);
#endif

    sparse_conv<conv2_iq_t, conv2_t, ap_uint<6>, conv2_weight_t, conv2_bias_t, conv2_accum_t, 20, 1, 3, 5>(layer7_out, layer8_out, sparse_hash_pool1, w8, b8); // conv2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<conv2_t>(layer8_out, "conv2", 60);
#endif

    sparse_relu<conv2_t, conv2_relu_t, 20, 3>(layer8_out, layer9_out); // conv2_relu
#ifndef __SYNTHESIS__
    nnet::save_layer_output<conv2_relu_t>(layer9_out, "conv2_relu", 60);
#endif

    ap_uint<6> sparse_hash_pool2[20 * 2];
#pragma HLS ARRAY_PARTITION variable=sparse_hash_pool2 complete dim=0
sparse_pooling_avg<conv2_relu_t, pool2_t, ap_uint<6>, pool2_accum_t, 20, 3, 2>(layer9_out, layer10_out, sparse_hash_pool1, sparse_hash_pool2); // pool2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<pool2_t>(layer10_out, "pool2", 60);
#endif

    sparse_flatten<pool2_t, flatten_t, ap_uint<6>, 6, 6, 3, 20>(layer10_out, sparse_hash_pool2, layer11_out); // flatten
#ifndef __SYNTHESIS__
    nnet::save_layer_output<flatten_t>(layer11_out, "flatten", 108);
#endif

    nnet::dense1_iq<flatten_t, dense1_iq_t>(layer11_out, layer12_out); // dense1_iq
#ifndef __SYNTHESIS__
    nnet::save_layer_output<dense1_iq_t>(layer12_out, "dense1_iq", 108);
#endif

    nnet::dense<dense1_iq_t, dense1_t, config13>(layer12_out, layer13_out, w13, b13); // dense1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<dense1_t>(layer13_out, "dense1", 36);
#endif

    nnet::relu<dense1_t, dense1_relu_t, relu_config14>(layer13_out, layer14_out); // dense1_relu
#ifndef __SYNTHESIS__
    nnet::save_layer_output<dense1_relu_t>(layer14_out, "dense1_relu", 36);
#endif

    nnet::dense2_iq<dense1_relu_t, dense2_iq_t>(layer14_out, layer15_out); // dense2_iq
#ifndef __SYNTHESIS__
    nnet::save_layer_output<dense2_iq_t>(layer15_out, "dense2_iq", 36);
#endif

    nnet::dense<dense2_iq_t, dense2_t, config16>(layer15_out, layer16_out, w16, b16); // dense2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<dense2_t>(layer16_out, "dense2", 10);
#endif

    nnet::softmax<dense2_t, result_t, softmax_config17>(layer16_out, layer17_out); // softmax
#ifndef __SYNTHESIS__
    nnet::save_layer_output<result_t>(layer17_out, "softmax", 10);
#endif

}

