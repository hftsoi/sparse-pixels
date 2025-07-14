#ifndef HLS_FULL_H_
#define HLS_FULL_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void hls_full(
    input_t x_in[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    result_t layer15_out[N_LAYER_13]
);


#endif
