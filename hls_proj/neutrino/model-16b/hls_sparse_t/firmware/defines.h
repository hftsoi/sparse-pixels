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
#define OUT_HEIGHT_2 63
#define OUT_WIDTH_2 63
#define N_FILT_2 1
#define OUT_HEIGHT_2 63
#define OUT_WIDTH_2 63
#define N_FILT_2 1
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

#define N_MAX_PIXELS 8

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<39,20> conv1_accum_t;
typedef ap_fixed<39,20> conv1_result_t;
typedef ap_fixed<16,7> weight2_t;
typedef ap_fixed<16,7> bias2_t;
typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<39,20> conv2_accum_t;
typedef ap_fixed<39,20> conv2_result_t;
typedef ap_fixed<16,7> weight5_t;
typedef ap_fixed<16,7> bias5_t;
typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> layer7_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef ap_ufixed<28,12> pool1_accum_t;
typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> layer8_t;
typedef ap_fixed<41,22> dense1_accum_t;
typedef ap_fixed<41,22> dense1_result_t;
typedef ap_fixed<16,7> weight10_t;
typedef ap_fixed<16,7> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> layer12_t;
typedef ap_fixed<18,8> relu3_table_t;
typedef ap_fixed<27,10> dense2_accum_t;
typedef ap_fixed<27,10> dense2_result_t;
typedef ap_fixed<16,7> weight13_t;
typedef ap_fixed<16,7> bias13_t;
typedef ap_uint<1> layer13_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> sigmoid_table_t;

typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> conv_default_t;
typedef ap_ufixed<16,6,AP_RND_CONV,AP_SAT,0> model_default_t;

#endif
