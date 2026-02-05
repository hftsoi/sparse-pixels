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
#define OUT_HEIGHT_2 22
#define OUT_WIDTH_2 22
#define N_FILT_2 8
#define OUT_HEIGHT_2 22
#define OUT_WIDTH_2 22
#define N_FILT_2 8
#define OUT_HEIGHT_4 10
#define OUT_WIDTH_4 10
#define N_FILT_4 8
#define OUT_HEIGHT_4 10
#define OUT_WIDTH_4 10
#define N_FILT_4 8
#define OUT_HEIGHT_6 8
#define OUT_WIDTH_6 8
#define N_FILT_6 8
#define OUT_HEIGHT_6 8
#define OUT_WIDTH_6 8
#define N_FILT_6 8
#define OUT_HEIGHT_8 4
#define OUT_WIDTH_8 4
#define N_FILT_8 4
#define OUT_HEIGHT_8 4
#define OUT_WIDTH_8 4
#define N_FILT_8 4
#define N_SIZE_0_10 64
#define N_LAYER_11 3


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<20,8> conv2d_accum_t;
typedef ap_fixed<20,8> conv2d_result_t;
typedef ap_fixed<16,6> conv2d_weight_t;
typedef ap_fixed<16,6> conv2d_bias_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<18,8> conv2d_relu_table_t;
typedef ap_fixed<20,8> conv2d_1_accum_t;
typedef ap_fixed<20,8> conv2d_1_result_t;
typedef ap_fixed<16,6> conv2d_1_weight_t;
typedef ap_fixed<16,6> conv2d_1_bias_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<18,8> conv2d_1_relu_table_t;
typedef ap_fixed<20,8> conv2d_2_accum_t;
typedef ap_fixed<20,8> conv2d_2_result_t;
typedef ap_fixed<16,6> conv2d_2_weight_t;
typedef ap_fixed<16,6> conv2d_2_bias_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<18,8> conv2d_2_relu_table_t;
typedef ap_fixed<20,8> conv2d_4_accum_t;
typedef ap_fixed<20,8> conv2d_4_result_t;
typedef ap_fixed<16,6> conv2d_4_weight_t;
typedef ap_fixed<16,6> conv2d_4_bias_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<18,8> conv2d_4_relu_table_t;
typedef ap_fixed<20,8> latent_z_mean_accum_t;
typedef ap_fixed<20,8> result_t;
typedef ap_fixed<16,6> latent_z_mean_weight_t;
typedef ap_fixed<16,6> latent_z_mean_bias_t;
typedef ap_uint<1> layer11_index;


#endif
