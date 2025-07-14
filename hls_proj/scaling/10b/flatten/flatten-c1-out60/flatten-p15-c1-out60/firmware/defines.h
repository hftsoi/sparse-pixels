#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 10
#define N_INPUT_2_1 6
#define N_INPUT_3_1 1
#define OUT_HEIGHT_2 10
#define OUT_WIDTH_2 6
#define N_FILT_2 1

#define N_MAX_PIXELS 15

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<10,2> input_t;
typedef ap_fixed<10,2> result_t;

typedef ap_fixed<27,12> conv1_accum_t;

typedef ap_fixed<10,2> weight2_t;
typedef ap_fixed<10,2> bias2_t;


#endif
