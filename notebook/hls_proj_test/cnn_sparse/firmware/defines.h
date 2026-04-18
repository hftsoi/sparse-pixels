#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<11,3> x_in_t;
typedef ap_fixed<11,3> input_reduce_t;
typedef ap_fixed<10,3> conv1_iq_t;
typedef ap_fixed<15,7> conv1_accum_t;
typedef ap_fixed<15,7> conv1_t;
typedef ap_fixed<4,3> conv1_weight_t;
typedef ap_ufixed<4,2> conv1_bias_t;
typedef ap_ufixed<14,6> conv1_relu_t;
typedef ap_ufixed<18,6> pool1_accum_t;
typedef ap_fixed<9,1> pool1_t;
typedef ap_fixed<8,1> conv2_iq_t;
typedef ap_fixed<13,6> conv2_accum_t;
typedef ap_fixed<13,6> conv2_t;
typedef ap_fixed<4,4> conv2_weight_t;
typedef ap_ufixed<3,0> conv2_bias_t;
typedef ap_ufixed<12,5> conv2_relu_t;
typedef ap_ufixed<14,5> pool2_accum_t;
typedef ap_ufixed<14,5> pool2_t;
typedef ap_fixed<12,4> flatten_t;
typedef ap_fixed<11,4> dense1_iq_t;
typedef ap_fixed<13,8> dense1_accum_t;
typedef ap_fixed<11,8> dense1_t;
typedef ap_fixed<6,3> dense1_weight_t;
typedef ap_fixed<5,1> dense1_bias_t;
typedef ap_uint<1> layer13_index;
typedef ap_fixed<8,5> dense1_relu_t;
typedef ap_fixed<18,8> dense1_relu_table_t;
typedef ap_fixed<7,5> dense2_iq_t;
typedef ap_fixed<14,9> dense2_accum_t;
typedef ap_fixed<14,9> dense2_t;
typedef ap_fixed<5,2> dense2_weight_t;
typedef ap_fixed<3,1> dense2_bias_t;
typedef ap_uint<1> layer16_index;
typedef ap_fixed<16,6> model_default_t;
typedef ap_ufixed<36,16> result_t;
typedef ap_fixed<18,8> softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_inv_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_inv_inp_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
