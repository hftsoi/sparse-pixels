#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 56
#define N_INPUT_2_1 56
#define N_INPUT_3_1 1
#define OUT_HEIGHT_2 56
#define OUT_WIDTH_2 56
#define N_FILT_2 3
#define OUT_HEIGHT_2 56
#define OUT_WIDTH_2 56
#define N_FILT_2 3
#define OUT_HEIGHT_5 56
#define OUT_WIDTH_5 56
#define N_FILT_5 1
#define OUT_HEIGHT_5 56
#define OUT_WIDTH_5 56
#define N_FILT_5 1
#define OUT_HEIGHT_8 14
#define OUT_WIDTH_8 14
#define N_FILT_8 1
#define N_SIZE_0_9 196
#define N_LAYER_10 20
#define N_LAYER_10 20
#define N_LAYER_13 5
#define N_LAYER_13 5

#define N_MAX_PIXELS 8

// hls-fpga-machine-learning insert layer-precision
typedef ap_ufixed<8,1> input_t;
typedef ap_fixed<24,14> conv1_accum_t;
typedef ap_fixed<24,14> conv1_result_t;
typedef ap_fixed<8,5> weight2_t;
typedef ap_fixed<8,5> bias2_t;
typedef ap_ufixed<8,4,AP_RND_CONV,AP_SAT,0> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<25,18> conv2_accum_t;
typedef ap_fixed<25,18> conv2_result_t;
typedef ap_fixed<8,5> weight5_t;
typedef ap_fixed<8,5> bias5_t;
typedef ap_ufixed<8,4,AP_RND_CONV,AP_SAT,0> layer7_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef ap_ufixed<16,8> pool1_accum_t;
typedef ap_ufixed<8,4,AP_RND_CONV,AP_SAT,0> layer8_t;
typedef ap_fixed<25,18> dense1_accum_t;
typedef ap_fixed<25,18> dense1_result_t;
typedef ap_fixed<8,5> weight10_t;
typedef ap_fixed<8,5> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_ufixed<8,4,AP_RND_CONV,AP_SAT,0> layer12_t;
typedef ap_fixed<18,8> relu3_table_t;
typedef ap_fixed<22,8> dense2_accum_t;
typedef ap_fixed<22,8> dense2_result_t;
typedef ap_fixed<8,5> weight13_t;
typedef ap_fixed<8,5> bias13_t;
typedef ap_uint<1> layer13_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_inv_table_t;

typedef ap_ufixed<8,4,AP_RND_CONV,AP_SAT,0> conv_default_t;
typedef ap_ufixed<8,4,AP_RND_CONV,AP_SAT,0> model_default_t;

#endif
