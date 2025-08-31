#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 63
#define N_INPUT_2_1 63
#define N_INPUT_3_1 1
#define OUT_HEIGHT_16 69
#define OUT_WIDTH_16 69
#define N_CHAN_16 1
#define OUT_HEIGHT_2 63
#define OUT_WIDTH_2 63
#define N_FILT_2 1
#define OUT_HEIGHT_2 63
#define OUT_WIDTH_2 63
#define N_FILT_2 1
#define OUT_HEIGHT_17 69
#define OUT_WIDTH_17 69
#define N_CHAN_17 1
#define OUT_HEIGHT_5 63
#define OUT_WIDTH_5 63
#define N_FILT_5 3
#define OUT_HEIGHT_5 63
#define OUT_WIDTH_5 63
#define N_FILT_5 3
#define OUT_HEIGHT_8 9
#define OUT_WIDTH_8 9
#define N_FILT_8 3
#define N_SIZE_0_9 243
#define N_LAYER_10 16
#define N_LAYER_10 16
#define N_LAYER_13 1
#define N_LAYER_13 1


// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_ufixed<8,1>, 1*1> input_t;
typedef nnet::array<ap_ufixed<8,1>, 1*1> layer16_t;
typedef ap_fixed<23,11> conv1_accum_t;
typedef nnet::array<ap_fixed<23,11>, 1*1> conv1_result_t;
typedef ap_fixed<8,3> weight2_t;
typedef ap_fixed<8,3> bias2_t;
typedef nnet::array<ap_ufixed<8,2,AP_RND_CONV,AP_SAT,0>, 1*1> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef nnet::array<ap_ufixed<8,2,AP_RND_CONV,AP_SAT,0>, 1*1> layer17_t;
typedef ap_fixed<23,12> conv2_accum_t;
typedef nnet::array<ap_fixed<23,12>, 3*1> conv2_result_t;
typedef ap_fixed<8,3> weight5_t;
typedef ap_fixed<8,3> bias5_t;
typedef nnet::array<ap_ufixed<8,2,AP_RND_CONV,AP_SAT,0>, 3*1> layer7_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef ap_ufixed<20,8> pool1_accum_t;
typedef nnet::array<ap_ufixed<8,2,AP_RND_CONV,AP_SAT,0>, 3*1> layer8_t;
typedef ap_fixed<25,14> dense1_accum_t;
typedef nnet::array<ap_fixed<25,14>, 16*1> dense1_result_t;
typedef ap_fixed<8,3> weight10_t;
typedef ap_fixed<8,3> bias10_t;
typedef ap_uint<1> layer10_index;
typedef nnet::array<ap_ufixed<8,2,AP_RND_CONV,AP_SAT,0>, 16*1> layer12_t;
typedef ap_fixed<18,8> relu3_table_t;
typedef ap_fixed<21,8> dense2_accum_t;
typedef nnet::array<ap_fixed<21,8>, 1*1> dense2_result_t;
typedef ap_fixed<8,3> weight13_t;
typedef ap_fixed<8,3> bias13_t;
typedef ap_uint<1> layer13_index;
typedef nnet::array<ap_fixed<16,6>, 1*1> result_t;
typedef ap_fixed<18,8> sigmoid_table_t;


#endif
