#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 64
#define N_INPUT_2_1 64
#define N_INPUT_3_1 1
#define OUT_HEIGHT_19 70
#define OUT_WIDTH_19 70
#define N_CHAN_19 1
#define OUT_HEIGHT_2 64
#define OUT_WIDTH_2 64
#define N_FILT_2 1
#define OUT_HEIGHT_2 64
#define OUT_WIDTH_2 64
#define N_FILT_2 1
#define OUT_HEIGHT_20 70
#define OUT_WIDTH_20 70
#define N_CHAN_20 1
#define OUT_HEIGHT_5 64
#define OUT_WIDTH_5 64
#define N_FILT_5 1
#define OUT_HEIGHT_5 64
#define OUT_WIDTH_5 64
#define N_FILT_5 1
#define OUT_HEIGHT_21 70
#define OUT_WIDTH_21 70
#define N_CHAN_21 1
#define OUT_HEIGHT_8 64
#define OUT_WIDTH_8 64
#define N_FILT_8 4
#define OUT_HEIGHT_8 64
#define OUT_WIDTH_8 64
#define N_FILT_8 4
#define OUT_HEIGHT_11 8
#define OUT_WIDTH_11 8
#define N_FILT_11 4
#define N_SIZE_0_12 256
#define N_LAYER_13 12
#define N_LAYER_13 12
#define N_LAYER_16 1
#define N_LAYER_16 1


// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_ufixed<8,1>, 1*1> input_t;
typedef nnet::array<ap_ufixed<8,1>, 1*1> layer19_t;
typedef ap_fixed<23,9> conv1_accum_t;
typedef nnet::array<ap_fixed<23,9>, 1*1> conv1_result_t;
typedef ap_fixed<8,1> weight2_t;
typedef ap_fixed<8,1> bias2_t;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0>, 1*1> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0>, 1*1> layer20_t;
typedef ap_fixed<23,8> conv2_accum_t;
typedef nnet::array<ap_fixed<23,8>, 1*1> conv2_result_t;
typedef ap_fixed<8,1> weight5_t;
typedef ap_fixed<8,1> bias5_t;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0>, 1*1> layer7_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0>, 1*1> layer21_t;
typedef ap_fixed<23,8> conv3_accum_t;
typedef nnet::array<ap_fixed<23,8>, 4*1> conv3_result_t;
typedef ap_fixed<8,1> weight8_t;
typedef ap_fixed<8,1> bias8_t;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0>, 4*1> layer10_t;
typedef ap_fixed<18,8> relu3_table_t;
typedef ap_ufixed<20,6> pool1_accum_t;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0>, 4*1> layer11_t;
typedef ap_fixed<25,10> dense1_accum_t;
typedef nnet::array<ap_fixed<25,10>, 12*1> dense1_result_t;
typedef ap_fixed<8,1> weight13_t;
typedef ap_fixed<8,1> bias13_t;
typedef ap_uint<1> layer13_index;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0>, 12*1> layer15_t;
typedef ap_fixed<18,8> relu4_table_t;
typedef ap_fixed<21,6> dense2_accum_t;
typedef nnet::array<ap_fixed<21,6>, 1*1> dense2_result_t;
typedef ap_fixed<8,1> weight16_t;
typedef ap_fixed<8,1> bias16_t;
typedef ap_uint<1> layer16_index;
typedef nnet::array<ap_fixed<16,6>, 1*1> result_t;
typedef ap_fixed<18,8> sigmoid_table_t;


#endif
