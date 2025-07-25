#ifndef MYHLS_H_
#define MYHLS_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void myhls(
    hls::stream<input_t> &x_in,
    hls::stream<result_t> &layer16_out
);


#endif
