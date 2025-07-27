#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 24
#define N_INPUT_2_1 24
#define N_INPUT_3_1 1
#define OUT_HEIGHT_19 28
#define OUT_WIDTH_19 28
#define N_CHAN_19 1
#define OUT_HEIGHT_2 24
#define OUT_WIDTH_2 24
#define N_FILT_2 3
#define OUT_HEIGHT_2 24
#define OUT_WIDTH_2 24
#define N_FILT_2 3
#define OUT_HEIGHT_20 28
#define OUT_WIDTH_20 28
#define N_CHAN_20 3
#define OUT_HEIGHT_5 24
#define OUT_WIDTH_5 24
#define N_FILT_5 2
#define OUT_HEIGHT_5 24
#define OUT_WIDTH_5 24
#define N_FILT_5 2
#define OUT_HEIGHT_21 28
#define OUT_WIDTH_21 28
#define N_CHAN_21 2
#define OUT_HEIGHT_8 24
#define OUT_WIDTH_8 24
#define N_FILT_8 1
#define OUT_HEIGHT_8 24
#define OUT_WIDTH_8 24
#define N_FILT_8 1
#define OUT_HEIGHT_11 12
#define OUT_WIDTH_11 12
#define N_FILT_11 1
#define N_SIZE_0_12 144
#define N_LAYER_13 28
#define N_LAYER_13 28
#define N_LAYER_16 5
#define N_LAYER_16 5


// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_ufixed<8,1>, 1*1> input_t;
typedef nnet::array<ap_ufixed<8,1>, 1*1> layer19_t;
typedef ap_fixed<22,9> conv1_accum_t;
typedef nnet::array<ap_fixed<22,9>, 3*1> conv1_result_t;
typedef ap_fixed<8,2> weight2_t;
typedef ap_fixed<8,2> bias2_t;
typedef nnet::array<ap_ufixed<8,1,AP_RND_CONV,AP_SAT,0>, 3*1> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef nnet::array<ap_ufixed<8,1,AP_RND_CONV,AP_SAT,0>, 3*1> layer20_t;
typedef ap_fixed<24,11> conv2_accum_t;
typedef nnet::array<ap_fixed<24,11>, 2*1> conv2_result_t;
typedef ap_fixed<8,2> weight5_t;
typedef ap_fixed<8,2> bias5_t;
typedef nnet::array<ap_ufixed<8,1,AP_RND_CONV,AP_SAT,0>, 2*1> layer7_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef nnet::array<ap_ufixed<8,1,AP_RND_CONV,AP_SAT,0>, 2*1> layer21_t;
typedef ap_fixed<23,10> conv3_accum_t;
typedef nnet::array<ap_fixed<23,10>, 1*1> conv3_result_t;
typedef ap_fixed<8,2> weight8_t;
typedef ap_fixed<8,2> bias8_t;
typedef nnet::array<ap_ufixed<8,1,AP_RND_CONV,AP_SAT,0>, 1*1> layer10_t;
typedef ap_fixed<18,8> relu3_table_t;
typedef ap_ufixed<12,3> pool1_accum_t;
typedef nnet::array<ap_ufixed<8,1,AP_RND_CONV,AP_SAT,0>, 1*1> layer11_t;
typedef ap_fixed<25,12> dense1_accum_t;
typedef nnet::array<ap_fixed<25,12>, 28*1> dense1_result_t;
typedef ap_fixed<8,2> weight13_t;
typedef ap_fixed<8,2> bias13_t;
typedef ap_uint<1> layer13_index;
typedef nnet::array<ap_ufixed<8,1,AP_RND_CONV,AP_SAT,0>, 28*1> layer15_t;
typedef ap_fixed<18,8> relu4_table_t;
typedef ap_fixed<22,9> dense2_accum_t;
typedef nnet::array<ap_fixed<22,9>, 5*1> dense2_result_t;
typedef ap_fixed<8,2> weight16_t;
typedef ap_fixed<8,2> bias16_t;
typedef ap_uint<1> layer16_index;
typedef nnet::array<ap_fixed<16,6>, 5*1> result_t;
typedef ap_fixed<18,8> softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_inv_table_t;


#endif
