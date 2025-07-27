#include <iostream>

#include "myhls.h"
#include "parameters.h"


void myhls(
    hls::stream<input_t> &x_in,
    hls::stream<result_t> &layer18_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=x_in,layer18_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<weight2_t, 75>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 3>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight5_t, 150>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 2>(b5, "b5.txt");
        nnet::load_weights_from_txt<weight8_t, 50>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 1>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight13_t, 4032>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 28>(b13, "b13.txt");
        nnet::load_weights_from_txt<weight16_t, 140>(w16, "w16.txt");
        nnet::load_weights_from_txt<bias16_t, 5>(b16, "b16.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=784
    nnet::zeropad2d_cl<input_t, layer19_t, config19>(x_in, layer19_out); // zp2d_conv1

    hls::stream<conv1_result_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=576
    nnet::conv_2d_cl<layer19_t, conv1_result_t, config2>(layer19_out, layer2_out, w2, b2); // conv1

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=576
    nnet::relu<conv1_result_t, layer4_t, relu_config4>(layer2_out, layer4_out); // relu1

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=784
    nnet::zeropad2d_cl<layer4_t, layer20_t, config20>(layer4_out, layer20_out); // zp2d_conv2

    hls::stream<conv2_result_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=576
    nnet::conv_2d_cl<layer20_t, conv2_result_t, config5>(layer20_out, layer5_out, w5, b5); // conv2

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=576
    nnet::relu<conv2_result_t, layer7_t, relu_config7>(layer5_out, layer7_out); // relu2

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=784
    nnet::zeropad2d_cl<layer7_t, layer21_t, config21>(layer7_out, layer21_out); // zp2d_conv3

    hls::stream<conv3_result_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=576
    nnet::conv_2d_cl<layer21_t, conv3_result_t, config8>(layer21_out, layer8_out, w8, b8); // conv3

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=576
    nnet::relu<conv3_result_t, layer10_t, relu_config10>(layer8_out, layer10_out); // relu3

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=144
    nnet::pooling2d_cl<layer10_t, layer11_t, config11>(layer10_out, layer11_out); // pool1

    auto& layer12_out = layer11_out;
    hls::stream<dense1_result_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=1
    nnet::dense<layer11_t, dense1_result_t, config13>(layer12_out, layer13_out, w13, b13); // dense1

    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=1
    nnet::relu<dense1_result_t, layer15_t, relu_config15>(layer13_out, layer15_out); // relu4

    hls::stream<dense2_result_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=1
    nnet::dense<layer15_t, dense2_result_t, config16>(layer15_out, layer16_out, w16, b16); // dense2

    nnet::softmax<dense2_result_t, result_t, softmax_config18>(layer16_out, layer18_out); // softmax

}

