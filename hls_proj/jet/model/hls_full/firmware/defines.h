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
#define N_FILT_2 3
#define OUT_HEIGHT_2 20
#define OUT_WIDTH_2 20
#define N_FILT_2 3
#define OUT_HEIGHT_5 10
#define OUT_WIDTH_5 10
#define N_FILT_5 3
#define OUT_HEIGHT_6 10
#define OUT_WIDTH_6 10
#define N_FILT_6 1
#define OUT_HEIGHT_6 10
#define OUT_WIDTH_6 10
#define N_FILT_6 1
#define N_SIZE_0_9 100
#define N_LAYER_10 64
#define N_LAYER_10 64
#define N_LAYER_13 5
#define N_LAYER_13 5


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<9,2> input_t;
typedef ap_fixed<29,12> conv1_accum_t;
typedef ap_fixed<29,12> conv1_result_t;
typedef ap_fixed<8,1> weight2_t;
typedef ap_fixed<8,1> bias2_t;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_ufixed<12,2> pool1_accum_t;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0> layer5_t;
typedef ap_fixed<22,7> conv2_accum_t;
typedef ap_fixed<22,7> conv2_result_t;
typedef ap_fixed<8,1> weight6_t;
typedef ap_fixed<8,1> bias6_t;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0> layer8_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef ap_fixed<24,9> dense1_accum_t;
typedef ap_fixed<24,9> dense1_result_t;
typedef ap_fixed<8,1> weight10_t;
typedef ap_fixed<8,1> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0> layer12_t;
typedef ap_fixed<18,8> relu3_table_t;
typedef ap_fixed<23,8> dense2_accum_t;
typedef ap_fixed<23,8> dense2_result_t;
typedef ap_fixed<8,1> weight13_t;
typedef ap_fixed<8,1> bias13_t;
typedef ap_uint<1> layer13_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_inv_table_t;


#endif
