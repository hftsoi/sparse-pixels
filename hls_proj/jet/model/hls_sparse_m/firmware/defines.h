#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 20
#define N_INPUT_2_1 20
#define N_INPUT_3_1 1
#define OUT_HEIGHT_2 20
#define OUT_WIDTH_2 20
#define N_FILT_2 2
#define OUT_HEIGHT_2 20
#define OUT_WIDTH_2 20
#define N_FILT_2 2
#define OUT_HEIGHT_5 20
#define OUT_WIDTH_5 20
#define N_FILT_5 2
#define OUT_HEIGHT_5 20
#define OUT_WIDTH_5 20
#define N_FILT_5 2
#define OUT_HEIGHT_8 10
#define OUT_WIDTH_8 10
#define N_FILT_8 2
#define OUT_HEIGHT_9 10
#define OUT_WIDTH_9 10
#define N_FILT_9 1
#define OUT_HEIGHT_9 10
#define OUT_WIDTH_9 10
#define N_FILT_9 1
#define N_SIZE_0_12 100
#define N_LAYER_13 64
#define N_LAYER_13 64
#define N_LAYER_16 5
#define N_LAYER_16 5

#define N_MAX_PIXELS 18

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<9,5> input_t;
typedef ap_fixed<27,12> conv1_accum_t;
typedef ap_fixed<27,12> conv1_result_t;
typedef ap_fixed<6,1> weight2_t;
typedef ap_fixed<6,1> bias2_t;
typedef ap_ufixed<6,0,AP_RND_CONV,AP_SAT,0> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<18,7> conv2_accum_t;
typedef ap_fixed<18,7> conv2_result_t;
typedef ap_fixed<6,1> weight5_t;
typedef ap_fixed<6,1> bias5_t;
typedef ap_ufixed<6,0,AP_RND_CONV,AP_SAT,0> layer7_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef ap_ufixed<10,2> pool1_accum_t;
typedef ap_ufixed<6,0,AP_RND_CONV,AP_SAT,0> layer8_t;
typedef ap_fixed<18,7> conv3_accum_t;
typedef ap_fixed<18,7> conv3_result_t;
typedef ap_fixed<6,1> weight9_t;
typedef ap_fixed<6,1> bias9_t;
typedef ap_ufixed<6,0,AP_RND_CONV,AP_SAT,0> layer11_t;
typedef ap_fixed<18,8> relu3_table_t;
typedef ap_fixed<20,9> dense1_accum_t;
typedef ap_fixed<20,9> dense1_result_t;
typedef ap_fixed<6,1> weight13_t;
typedef ap_fixed<6,1> bias13_t;
typedef ap_uint<1> layer13_index;
typedef ap_ufixed<6,0,AP_RND_CONV,AP_SAT,0> layer15_t;
typedef ap_fixed<18,8> relu4_table_t;
typedef ap_fixed<19,8> dense2_accum_t;
typedef ap_fixed<19,8> dense2_result_t;
typedef ap_fixed<6,1> weight16_t;
typedef ap_fixed<6,1> bias16_t;
typedef ap_uint<1> layer16_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_inv_table_t;

typedef ap_ufixed<9,2,AP_RND_CONV,AP_SAT,0> conv_default_t;
typedef ap_ufixed<6,2,AP_RND_CONV,AP_SAT,0> model_default_t;

#endif
