#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    x_in_t x_in[48*48*1],
    result_t layer16_out[10]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x_in complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x_in,layer16_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<conv1_weight_t, 49>(w3, "w3.txt");
        nnet::load_weights_from_txt<conv1_bias_t, 1>(b3, "b3.txt");
        nnet::load_weights_from_txt<conv2_weight_t, 75>(w7, "w7.txt");
        nnet::load_weights_from_txt<conv2_bias_t, 3>(b7, "b7.txt");
        nnet::load_weights_from_txt<dense1_weight_t, 3888>(w12, "w12.txt");
        nnet::load_weights_from_txt<dense1_bias_t, 36>(b12, "b12.txt");
        nnet::load_weights_from_txt<dense2_weight_t, 360>(w15, "w15.txt");
        nnet::load_weights_from_txt<dense2_bias_t, 10>(b15, "b15.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    conv1_iq_t layer2_out[48*48*1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    conv1_t layer3_out[48*48*1];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    conv1_relu_t layer4_out[48*48*1];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    pool1_t layer5_out[12*12*1];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    conv2_iq_t layer6_out[12*12*1];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    conv2_t layer7_out[12*12*3];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    conv2_relu_t layer8_out[12*12*3];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    pool2_t layer9_out[6*6*3];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    auto& layer10_out = layer9_out;
    dense1_iq_t layer11_out[108];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0

    dense1_t layer12_out[36];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    dense1_relu_t layer13_out[36];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0

    dense2_iq_t layer14_out[36];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0

    dense2_t layer15_out[10];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0

    nnet::conv1_iq<x_in_t, conv1_iq_t>(x_in, layer2_out); // conv1_iq
#ifndef __SYNTHESIS__
    nnet::save_layer_output<conv1_iq_t>(layer2_out, "conv1_iq", 48*48*1);
#endif

    nnet::conv_2d_cl<conv1_iq_t, conv1_t, config3>(layer2_out, layer3_out, w3, b3); // conv1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<conv1_t>(layer3_out, "conv1", 48*48*1);
#endif

    nnet::relu<conv1_t, conv1_relu_t, relu_config4>(layer3_out, layer4_out); // conv1_relu
#ifndef __SYNTHESIS__
    nnet::save_layer_output<conv1_relu_t>(layer4_out, "conv1_relu", 48*48*1);
#endif

    nnet::pooling2d_cl<conv1_relu_t, pool1_t, config5>(layer4_out, layer5_out); // pool1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<pool1_t>(layer5_out, "pool1", 12*12*1);
#endif

    nnet::conv2_iq<pool1_t, conv2_iq_t>(layer5_out, layer6_out); // conv2_iq
#ifndef __SYNTHESIS__
    nnet::save_layer_output<conv2_iq_t>(layer6_out, "conv2_iq", 12*12*1);
#endif

    nnet::conv_2d_cl<conv2_iq_t, conv2_t, config7>(layer6_out, layer7_out, w7, b7); // conv2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<conv2_t>(layer7_out, "conv2", 12*12*3);
#endif

    nnet::relu<conv2_t, conv2_relu_t, relu_config8>(layer7_out, layer8_out); // conv2_relu
#ifndef __SYNTHESIS__
    nnet::save_layer_output<conv2_relu_t>(layer8_out, "conv2_relu", 12*12*3);
#endif

    nnet::pooling2d_cl<conv2_relu_t, pool2_t, config9>(layer8_out, layer9_out); // pool2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<pool2_t>(layer9_out, "pool2", 6*6*3);
#endif

    nnet::dense1_iq<pool2_t, dense1_iq_t>(layer10_out, layer11_out); // dense1_iq
#ifndef __SYNTHESIS__
    nnet::save_layer_output<dense1_iq_t>(layer11_out, "dense1_iq", 108);
#endif

    nnet::dense<dense1_iq_t, dense1_t, config12>(layer11_out, layer12_out, w12, b12); // dense1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<dense1_t>(layer12_out, "dense1", 36);
#endif

    nnet::relu<dense1_t, dense1_relu_t, relu_config13>(layer12_out, layer13_out); // dense1_relu
#ifndef __SYNTHESIS__
    nnet::save_layer_output<dense1_relu_t>(layer13_out, "dense1_relu", 36);
#endif

    nnet::dense2_iq<dense1_relu_t, dense2_iq_t>(layer13_out, layer14_out); // dense2_iq
#ifndef __SYNTHESIS__
    nnet::save_layer_output<dense2_iq_t>(layer14_out, "dense2_iq", 36);
#endif

    nnet::dense<dense2_iq_t, dense2_t, config15>(layer14_out, layer15_out, w15, b15); // dense2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<dense2_t>(layer15_out, "dense2", 10);
#endif

    nnet::softmax<dense2_t, result_t, softmax_config16>(layer15_out, layer16_out); // softmax
#ifndef __SYNTHESIS__
    nnet::save_layer_output<result_t>(layer16_out, "softmax", 10);
#endif

}

