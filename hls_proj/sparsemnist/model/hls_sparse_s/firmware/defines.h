#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 40
#define N_INPUT_2_1 40
#define N_INPUT_3_1 1
#define OUT_HEIGHT_2 40
#define OUT_WIDTH_2 40
#define N_FILT_2 1
#define OUT_HEIGHT_2 40
#define OUT_WIDTH_2 40
#define N_FILT_2 1
#define OUT_HEIGHT_5 10
#define OUT_WIDTH_5 10
#define N_FILT_5 1
#define OUT_HEIGHT_6 10
#define OUT_WIDTH_6 10
#define N_FILT_6 3
#define OUT_HEIGHT_6 10
#define OUT_WIDTH_6 10
#define N_FILT_6 3
#define OUT_HEIGHT_9 5
#define OUT_WIDTH_9 5
#define N_FILT_9 3
#define N_SIZE_0_10 75
#define N_LAYER_11 64
#define N_LAYER_11 64
#define N_LAYER_14 10
#define N_LAYER_14 10

#define N_MAX_PIXELS 12

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<9,2> input_t;
typedef ap_fixed<27,12> conv1_accum_t;
typedef ap_fixed<27,12> conv1_result_t;
typedef ap_fixed<6,1> weight2_t;
typedef ap_fixed<6,1> bias2_t;
typedef ap_ufixed<6,0,AP_RND_CONV,AP_SAT,0> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_ufixed<14,4> pool1_accum_t;
typedef ap_ufixed<6,0,AP_RND_CONV,AP_SAT,0> layer5_t;
typedef ap_fixed<17,6> conv2_accum_t;
typedef ap_fixed<17,6> conv2_result_t;
typedef ap_fixed<6,1> weight6_t;
typedef ap_fixed<6,1> bias6_t;
typedef ap_ufixed<6,0,AP_RND_CONV,AP_SAT,0> layer8_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef ap_ufixed<10,2> pool2_accum_t;
typedef ap_ufixed<6,0,AP_RND_CONV,AP_SAT,0> layer9_t;
typedef ap_fixed<20,9> dense1_accum_t;
typedef ap_fixed<20,9> dense1_result_t;
typedef ap_fixed<6,1> weight11_t;
typedef ap_fixed<6,1> bias11_t;
typedef ap_uint<1> layer11_index;
typedef ap_ufixed<6,0,AP_RND_CONV,AP_SAT,0> layer13_t;
typedef ap_fixed<18,8> relu3_table_t;
typedef ap_fixed<19,8> dense2_accum_t;
typedef ap_fixed<19,8> dense2_result_t;
typedef ap_fixed<6,1> weight14_t;
typedef ap_fixed<6,1> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_inv_table_t;

typedef ap_ufixed<9,2,AP_RND_CONV,AP_SAT,0> conv_default_t;
typedef ap_ufixed<6,2,AP_RND_CONV,AP_SAT,0> model_default_t;

#endif
