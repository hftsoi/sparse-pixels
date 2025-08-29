#include <iostream>

#include "myhls.h"
#include "parameters.h"


void myhls(
    hls::stream<input_t> &x_in,
    hls::stream<result_t> &layer15_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=x_in,layer15_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<weight2_t, 243>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 3>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight5_t, 243>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 1>(b5, "b5.txt");
        nnet::load_weights_from_txt<weight10_t, 3920>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 20>(b10, "b10.txt");
        nnet::load_weights_from_txt<weight13_t, 100>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 5>(b13, "b13.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=4096
    nnet::zeropad2d_cl<input_t, layer16_t, config16>(x_in, layer16_out); // zp2d_conv1

    hls::stream<conv1_result_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=3136
    nnet::conv_2d_cl<layer16_t, conv1_result_t, config2>(layer16_out, layer2_out, w2, b2); // conv1

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=3136
    nnet::relu<conv1_result_t, layer4_t, relu_config4>(layer2_out, layer4_out); // relu1

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=4096
    nnet::zeropad2d_cl<layer4_t, layer17_t, config17>(layer4_out, layer17_out); // zp2d_conv2

    hls::stream<conv2_result_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=3136
    nnet::conv_2d_cl<layer17_t, conv2_result_t, config5>(layer17_out, layer5_out, w5, b5); // conv2

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=3136
    nnet::relu<conv2_result_t, layer7_t, relu_config7>(layer5_out, layer7_out); // relu2

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=196
    nnet::pooling2d_cl<layer7_t, layer8_t, config8>(layer7_out, layer8_out); // pool1

    auto& layer9_out = layer8_out;
    hls::stream<dense1_result_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=1
    nnet::dense<layer8_t, dense1_result_t, config10>(layer9_out, layer10_out, w10, b10); // dense1

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=1
    nnet::relu<dense1_result_t, layer12_t, relu_config12>(layer10_out, layer12_out); // relu3

    hls::stream<dense2_result_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=1
    nnet::dense<layer12_t, dense2_result_t, config13>(layer12_out, layer13_out, w13, b13); // dense2

    nnet::softmax<dense2_result_t, result_t, softmax_config15>(layer13_out, layer15_out); // softmax

}

