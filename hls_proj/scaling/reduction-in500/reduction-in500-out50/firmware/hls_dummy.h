#ifndef HLS_DUMMY_H_
#define HLS_DUMMY_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void hls_dummy(
    input_t x_in[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    result_t layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2]
);


#endif
