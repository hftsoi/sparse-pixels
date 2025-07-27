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
#define OUT_HEIGHT_2 24
#define OUT_WIDTH_2 24
#define N_FILT_2 3
#define OUT_HEIGHT_2 24
#define OUT_WIDTH_2 24
#define N_FILT_2 3
#define OUT_HEIGHT_5 24
#define OUT_WIDTH_5 24
#define N_FILT_5 2
#define OUT_HEIGHT_5 24
#define OUT_WIDTH_5 24
#define N_FILT_5 2
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

#define N_MAX_PIXELS 12

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<38,19> conv1_accum_t;
typedef ap_fixed<38,19> conv1_result_t;
typedef ap_fixed<16,7> weight2_t;
typedef ap_fixed<16,7> bias2_t;
typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<40,21> conv2_accum_t;
typedef ap_fixed<40,21> conv2_result_t;
typedef ap_fixed<16,7> weight5_t;
typedef ap_fixed<16,7> bias5_t;
typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> layer7_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef ap_fixed<39,20> conv3_accum_t;
typedef ap_fixed<39,20> conv3_result_t;
typedef ap_fixed<16,7> weight8_t;
typedef ap_fixed<16,7> bias8_t;
typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> layer10_t;
typedef ap_fixed<18,8> relu3_table_t;
typedef ap_ufixed<20,8> pool1_accum_t;
typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> layer11_t;
typedef ap_fixed<41,22> dense1_accum_t;
typedef ap_fixed<41,22> dense1_result_t;
typedef ap_fixed<16,7> weight13_t;
typedef ap_fixed<16,7> bias13_t;
typedef ap_uint<1> layer13_index;
typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> layer15_t;
typedef ap_fixed<18,8> relu4_table_t;
typedef ap_fixed<28,9> dense2_accum_t;
typedef ap_fixed<28,9> dense2_result_t;
typedef ap_fixed<16,7> weight16_t;
typedef ap_fixed<16,7> bias16_t;
typedef ap_uint<1> layer16_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_inv_table_t;

typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> conv_default_t;
typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> model_default_t;

#endif
