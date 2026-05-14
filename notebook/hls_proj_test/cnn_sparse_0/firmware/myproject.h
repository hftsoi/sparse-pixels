#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void myproject(
    x_in_t x_in[48*48*1],
    result_t layer17_out[10]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
