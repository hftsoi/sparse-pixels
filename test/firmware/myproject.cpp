#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_1[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    result_t layer11_out[N_LAYER_11]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer11_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<conv2d_weight_t, 72>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv2d_bias_t, 8>(b2, "b2.txt");
        nnet::load_weights_from_txt<conv2d_1_weight_t, 1024>(w4, "w4.txt");
        nnet::load_weights_from_txt<conv2d_1_bias_t, 8>(b4, "b4.txt");
        nnet::load_weights_from_txt<conv2d_2_weight_t, 576>(w6, "w6.txt");
        nnet::load_weights_from_txt<conv2d_2_bias_t, 8>(b6, "b6.txt");
        nnet::load_weights_from_txt<conv2d_4_weight_t, 128>(w8, "w8.txt");
        nnet::load_weights_from_txt<conv2d_4_bias_t, 4>(b8, "b8.txt");
        nnet::load_weights_from_txt<latent_z_mean_weight_t, 192>(w11, "w11.txt");
        nnet::load_weights_from_txt<latent_z_mean_bias_t, 3>(b11, "b11.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    conv2d_result_t layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::conv_2d_cl<input_t, conv2d_result_t, config2>(input_1, layer2_out, w2, b2); // conv2d

    layer3_t layer3_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::relu<conv2d_result_t, layer3_t, relu_config3>(layer2_out, layer3_out); // conv2d_relu

    conv2d_1_result_t layer4_out[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::conv_2d_cl<layer3_t, conv2d_1_result_t, config4>(layer3_out, layer4_out, w4, b4); // conv2d_1

    layer5_t layer5_out[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<conv2d_1_result_t, layer5_t, relu_config5>(layer4_out, layer5_out); // conv2d_1_relu

    conv2d_2_result_t layer6_out[OUT_HEIGHT_6*OUT_WIDTH_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::conv_2d_cl<layer5_t, conv2d_2_result_t, config6>(layer5_out, layer6_out, w6, b6); // conv2d_2

    layer7_t layer7_out[OUT_HEIGHT_6*OUT_WIDTH_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<conv2d_2_result_t, layer7_t, relu_config7>(layer6_out, layer7_out); // conv2d_2_relu

    conv2d_4_result_t layer8_out[OUT_HEIGHT_8*OUT_WIDTH_8*N_FILT_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::conv_2d_cl<layer7_t, conv2d_4_result_t, config8>(layer7_out, layer8_out, w8, b8); // conv2d_4

    layer9_t layer9_out[OUT_HEIGHT_8*OUT_WIDTH_8*N_FILT_8];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::relu<conv2d_4_result_t, layer9_t, relu_config9>(layer8_out, layer9_out); // conv2d_4_relu

    auto& layer10_out = layer9_out;
    nnet::dense<layer9_t, result_t, config11>(layer10_out, layer11_out, w11, b11); // latent_z_mean

}

