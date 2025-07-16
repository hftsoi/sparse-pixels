#include <iostream>

#include "myhls.h"
#include "parameters.h"


void myhls(
    hls::stream<input_t> &x_in,
    hls::stream<result_t> &layer16_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=x_in,layer16_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<weight2_t, 9>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 1>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight6_t, 27>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 3>(b6, "b6.txt");
        nnet::load_weights_from_txt<weight11_t, 4800>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 64>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight14_t, 640>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 10>(b14, "b14.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=1764
    nnet::zeropad2d_cl<input_t, layer17_t, config17>(x_in, layer17_out); // zp2d_conv1

    hls::stream<conv1_result_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=1600
    nnet::conv_2d_cl<layer17_t, conv1_result_t, config2>(layer17_out, layer2_out, w2, b2); // conv1

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=1600
    nnet::relu<conv1_result_t, layer4_t, relu_config4>(layer2_out, layer4_out); // relu1

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=100
    nnet::pooling2d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out); // pool1

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=144
    nnet::zeropad2d_cl<layer5_t, layer18_t, config18>(layer5_out, layer18_out); // zp2d_conv2

    hls::stream<conv2_result_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=100
    nnet::conv_2d_cl<layer18_t, conv2_result_t, config6>(layer18_out, layer6_out, w6, b6); // conv2

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=100
    nnet::relu<conv2_result_t, layer8_t, relu_config8>(layer6_out, layer8_out); // relu2

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=25
    nnet::pooling2d_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out); // pool2

    auto& layer10_out = layer9_out;
    hls::stream<dense1_result_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=1
    nnet::dense<layer9_t, dense1_result_t, config11>(layer10_out, layer11_out, w11, b11); // dense1

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=1
    nnet::relu<dense1_result_t, layer13_t, relu_config13>(layer11_out, layer13_out); // relu3

    hls::stream<dense2_result_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=1
    nnet::dense<layer13_t, dense2_result_t, config14>(layer13_out, layer14_out, w14, b14); // dense2

    nnet::softmax<dense2_result_t, result_t, softmax_config16>(layer14_out, layer16_out); // softmax

}

