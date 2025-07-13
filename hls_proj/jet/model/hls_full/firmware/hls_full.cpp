#include <iostream>

#include "hls_full.h"
#include "parameters.h"


void hls_full(
    input_t x_in[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    result_t layer18_out[N_LAYER_16]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x_in complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x_in,layer18_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<weight2_t, 18>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 2>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight5_t, 36>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 2>(b5, "b5.txt");
        nnet::load_weights_from_txt<weight9_t, 18>(w9, "w9.txt");
        nnet::load_weights_from_txt<bias9_t, 1>(b9, "b9.txt");
        nnet::load_weights_from_txt<weight13_t, 6400>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 64>(b13, "b13.txt");
        nnet::load_weights_from_txt<weight16_t, 320>(w16, "w16.txt");
        nnet::load_weights_from_txt<bias16_t, 5>(b16, "b16.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    conv1_result_t layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::conv_2d_cl<input_t, conv1_result_t, config2>(x_in, layer2_out, w2, b2); // conv1

    layer4_t layer4_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::relu<conv1_result_t, layer4_t, relu_config4>(layer2_out, layer4_out); // relu1

    conv2_result_t layer5_out[OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::conv_2d_cl<layer4_t, conv2_result_t, config5>(layer4_out, layer5_out, w5, b5); // conv2

    layer7_t layer7_out[OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<conv2_result_t, layer7_t, relu_config7>(layer5_out, layer7_out); // relu2

    layer8_t layer8_out[OUT_HEIGHT_8*OUT_WIDTH_8*N_FILT_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::pooling2d_cl<layer7_t, layer8_t, config8>(layer7_out, layer8_out); // pool1

    conv3_result_t layer9_out[OUT_HEIGHT_9*OUT_WIDTH_9*N_FILT_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::conv_2d_cl<layer8_t, conv3_result_t, config9>(layer8_out, layer9_out, w9, b9); // conv3

    layer11_t layer11_out[OUT_HEIGHT_9*OUT_WIDTH_9*N_FILT_9];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::relu<conv3_result_t, layer11_t, relu_config11>(layer9_out, layer11_out); // relu3

    auto& layer12_out = layer11_out;
    dense1_result_t layer13_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::dense<layer11_t, dense1_result_t, config13>(layer12_out, layer13_out, w13, b13); // dense1

    layer15_t layer15_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::relu<dense1_result_t, layer15_t, relu_config15>(layer13_out, layer15_out); // relu4

    dense2_result_t layer16_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::dense<layer15_t, dense2_result_t, config16>(layer15_out, layer16_out, w16, b16); // dense2

    nnet::softmax<dense2_result_t, result_t, softmax_config18>(layer16_out, layer18_out); // softmax

}

