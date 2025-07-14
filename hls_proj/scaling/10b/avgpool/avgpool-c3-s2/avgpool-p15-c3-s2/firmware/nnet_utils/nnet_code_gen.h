#ifndef NNET_INSTR_GEN_H_
#define NNET_INSTR_GEN_H_

#include "nnet_conv1d_latency.h"
#include "nnet_helpers.h"

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_function_stubs.h"
#include "nnet_mult.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T> class PointwiseConv1D {
  public:
    static void pointwise_conv(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                               res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                               typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                               typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
        // To be implemented in subclasses
    }
};

// hls4ml insert code
template<class data_T, typename CONFIG_T>
class fill_buffer_2 : public nnet::FillConv2DBuffer<data_T, CONFIG_T> {
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        if (partition ==   0) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =          0; buffer[0][4] =    data[0]; buffer[0][5] =    data[1]; buffer[0][6] =          0; buffer[0][7] =   data[20]; buffer[0][8] =   data[21];

        }
        if (partition ==   1) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =    data[0]; buffer[0][4] =    data[1]; buffer[0][5] =    data[2]; buffer[0][6] =   data[20]; buffer[0][7] =   data[21]; buffer[0][8] =   data[22];

        }
        if (partition ==   2) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =    data[1]; buffer[0][4] =    data[2]; buffer[0][5] =    data[3]; buffer[0][6] =   data[21]; buffer[0][7] =   data[22]; buffer[0][8] =   data[23];

        }
        if (partition ==   3) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =    data[2]; buffer[0][4] =    data[3]; buffer[0][5] =    data[4]; buffer[0][6] =   data[22]; buffer[0][7] =   data[23]; buffer[0][8] =   data[24];

        }
        if (partition ==   4) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =    data[3]; buffer[0][4] =    data[4]; buffer[0][5] =    data[5]; buffer[0][6] =   data[23]; buffer[0][7] =   data[24]; buffer[0][8] =   data[25];

        }
        if (partition ==   5) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =    data[4]; buffer[0][4] =    data[5]; buffer[0][5] =    data[6]; buffer[0][6] =   data[24]; buffer[0][7] =   data[25]; buffer[0][8] =   data[26];

        }
        if (partition ==   6) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =    data[5]; buffer[0][4] =    data[6]; buffer[0][5] =    data[7]; buffer[0][6] =   data[25]; buffer[0][7] =   data[26]; buffer[0][8] =   data[27];

        }
        if (partition ==   7) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =    data[6]; buffer[0][4] =    data[7]; buffer[0][5] =    data[8]; buffer[0][6] =   data[26]; buffer[0][7] =   data[27]; buffer[0][8] =   data[28];

        }
        if (partition ==   8) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =    data[7]; buffer[0][4] =    data[8]; buffer[0][5] =    data[9]; buffer[0][6] =   data[27]; buffer[0][7] =   data[28]; buffer[0][8] =   data[29];

        }
        if (partition ==   9) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =    data[8]; buffer[0][4] =    data[9]; buffer[0][5] =   data[10]; buffer[0][6] =   data[28]; buffer[0][7] =   data[29]; buffer[0][8] =   data[30];

        }
        if (partition ==  10) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =    data[9]; buffer[0][4] =   data[10]; buffer[0][5] =   data[11]; buffer[0][6] =   data[29]; buffer[0][7] =   data[30]; buffer[0][8] =   data[31];

        }
        if (partition ==  11) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =   data[10]; buffer[0][4] =   data[11]; buffer[0][5] =   data[12]; buffer[0][6] =   data[30]; buffer[0][7] =   data[31]; buffer[0][8] =   data[32];

        }
        if (partition ==  12) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =   data[11]; buffer[0][4] =   data[12]; buffer[0][5] =   data[13]; buffer[0][6] =   data[31]; buffer[0][7] =   data[32]; buffer[0][8] =   data[33];

        }
        if (partition ==  13) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =   data[12]; buffer[0][4] =   data[13]; buffer[0][5] =   data[14]; buffer[0][6] =   data[32]; buffer[0][7] =   data[33]; buffer[0][8] =   data[34];

        }
        if (partition ==  14) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =   data[13]; buffer[0][4] =   data[14]; buffer[0][5] =   data[15]; buffer[0][6] =   data[33]; buffer[0][7] =   data[34]; buffer[0][8] =   data[35];

        }
        if (partition ==  15) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =   data[14]; buffer[0][4] =   data[15]; buffer[0][5] =   data[16]; buffer[0][6] =   data[34]; buffer[0][7] =   data[35]; buffer[0][8] =   data[36];

        }
        if (partition ==  16) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =   data[15]; buffer[0][4] =   data[16]; buffer[0][5] =   data[17]; buffer[0][6] =   data[35]; buffer[0][7] =   data[36]; buffer[0][8] =   data[37];

        }
        if (partition ==  17) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =   data[16]; buffer[0][4] =   data[17]; buffer[0][5] =   data[18]; buffer[0][6] =   data[36]; buffer[0][7] =   data[37]; buffer[0][8] =   data[38];

        }
        if (partition ==  18) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =   data[17]; buffer[0][4] =   data[18]; buffer[0][5] =   data[19]; buffer[0][6] =   data[37]; buffer[0][7] =   data[38]; buffer[0][8] =   data[39];

        }
        if (partition ==  19) {
            buffer[0][0] =          0; buffer[0][1] =          0; buffer[0][2] =          0; buffer[0][3] =   data[18]; buffer[0][4] =   data[19]; buffer[0][5] =          0; buffer[0][6] =   data[38]; buffer[0][7] =   data[39]; buffer[0][8] =          0;

        }
        if (partition ==  20) {
            buffer[0][0] =          0; buffer[0][1] =    data[0]; buffer[0][2] =    data[1]; buffer[0][3] =          0; buffer[0][4] =   data[20]; buffer[0][5] =   data[21]; buffer[0][6] =          0; buffer[0][7] =   data[40]; buffer[0][8] =   data[41];

        }
        if (partition ==  21) {
            buffer[0][0] =    data[0]; buffer[0][1] =    data[1]; buffer[0][2] =    data[2]; buffer[0][3] =   data[20]; buffer[0][4] =   data[21]; buffer[0][5] =   data[22]; buffer[0][6] =   data[40]; buffer[0][7] =   data[41]; buffer[0][8] =   data[42];

        }
        if (partition ==  22) {
            buffer[0][0] =    data[1]; buffer[0][1] =    data[2]; buffer[0][2] =    data[3]; buffer[0][3] =   data[21]; buffer[0][4] =   data[22]; buffer[0][5] =   data[23]; buffer[0][6] =   data[41]; buffer[0][7] =   data[42]; buffer[0][8] =   data[43];

        }
        if (partition ==  23) {
            buffer[0][0] =    data[2]; buffer[0][1] =    data[3]; buffer[0][2] =    data[4]; buffer[0][3] =   data[22]; buffer[0][4] =   data[23]; buffer[0][5] =   data[24]; buffer[0][6] =   data[42]; buffer[0][7] =   data[43]; buffer[0][8] =   data[44];

        }
        if (partition ==  24) {
            buffer[0][0] =    data[3]; buffer[0][1] =    data[4]; buffer[0][2] =    data[5]; buffer[0][3] =   data[23]; buffer[0][4] =   data[24]; buffer[0][5] =   data[25]; buffer[0][6] =   data[43]; buffer[0][7] =   data[44]; buffer[0][8] =   data[45];

        }
        if (partition ==  25) {
            buffer[0][0] =    data[4]; buffer[0][1] =    data[5]; buffer[0][2] =    data[6]; buffer[0][3] =   data[24]; buffer[0][4] =   data[25]; buffer[0][5] =   data[26]; buffer[0][6] =   data[44]; buffer[0][7] =   data[45]; buffer[0][8] =   data[46];

        }
        if (partition ==  26) {
            buffer[0][0] =    data[5]; buffer[0][1] =    data[6]; buffer[0][2] =    data[7]; buffer[0][3] =   data[25]; buffer[0][4] =   data[26]; buffer[0][5] =   data[27]; buffer[0][6] =   data[45]; buffer[0][7] =   data[46]; buffer[0][8] =   data[47];

        }
        if (partition ==  27) {
            buffer[0][0] =    data[6]; buffer[0][1] =    data[7]; buffer[0][2] =    data[8]; buffer[0][3] =   data[26]; buffer[0][4] =   data[27]; buffer[0][5] =   data[28]; buffer[0][6] =   data[46]; buffer[0][7] =   data[47]; buffer[0][8] =   data[48];

        }
        if (partition ==  28) {
            buffer[0][0] =    data[7]; buffer[0][1] =    data[8]; buffer[0][2] =    data[9]; buffer[0][3] =   data[27]; buffer[0][4] =   data[28]; buffer[0][5] =   data[29]; buffer[0][6] =   data[47]; buffer[0][7] =   data[48]; buffer[0][8] =   data[49];

        }
        if (partition ==  29) {
            buffer[0][0] =    data[8]; buffer[0][1] =    data[9]; buffer[0][2] =   data[10]; buffer[0][3] =   data[28]; buffer[0][4] =   data[29]; buffer[0][5] =   data[30]; buffer[0][6] =   data[48]; buffer[0][7] =   data[49]; buffer[0][8] =   data[50];

        }
        if (partition ==  30) {
            buffer[0][0] =    data[9]; buffer[0][1] =   data[10]; buffer[0][2] =   data[11]; buffer[0][3] =   data[29]; buffer[0][4] =   data[30]; buffer[0][5] =   data[31]; buffer[0][6] =   data[49]; buffer[0][7] =   data[50]; buffer[0][8] =   data[51];

        }
        if (partition ==  31) {
            buffer[0][0] =   data[10]; buffer[0][1] =   data[11]; buffer[0][2] =   data[12]; buffer[0][3] =   data[30]; buffer[0][4] =   data[31]; buffer[0][5] =   data[32]; buffer[0][6] =   data[50]; buffer[0][7] =   data[51]; buffer[0][8] =   data[52];

        }
        if (partition ==  32) {
            buffer[0][0] =   data[11]; buffer[0][1] =   data[12]; buffer[0][2] =   data[13]; buffer[0][3] =   data[31]; buffer[0][4] =   data[32]; buffer[0][5] =   data[33]; buffer[0][6] =   data[51]; buffer[0][7] =   data[52]; buffer[0][8] =   data[53];

        }
        if (partition ==  33) {
            buffer[0][0] =   data[12]; buffer[0][1] =   data[13]; buffer[0][2] =   data[14]; buffer[0][3] =   data[32]; buffer[0][4] =   data[33]; buffer[0][5] =   data[34]; buffer[0][6] =   data[52]; buffer[0][7] =   data[53]; buffer[0][8] =   data[54];

        }
        if (partition ==  34) {
            buffer[0][0] =   data[13]; buffer[0][1] =   data[14]; buffer[0][2] =   data[15]; buffer[0][3] =   data[33]; buffer[0][4] =   data[34]; buffer[0][5] =   data[35]; buffer[0][6] =   data[53]; buffer[0][7] =   data[54]; buffer[0][8] =   data[55];

        }
        if (partition ==  35) {
            buffer[0][0] =   data[14]; buffer[0][1] =   data[15]; buffer[0][2] =   data[16]; buffer[0][3] =   data[34]; buffer[0][4] =   data[35]; buffer[0][5] =   data[36]; buffer[0][6] =   data[54]; buffer[0][7] =   data[55]; buffer[0][8] =   data[56];

        }
        if (partition ==  36) {
            buffer[0][0] =   data[15]; buffer[0][1] =   data[16]; buffer[0][2] =   data[17]; buffer[0][3] =   data[35]; buffer[0][4] =   data[36]; buffer[0][5] =   data[37]; buffer[0][6] =   data[55]; buffer[0][7] =   data[56]; buffer[0][8] =   data[57];

        }
        if (partition ==  37) {
            buffer[0][0] =   data[16]; buffer[0][1] =   data[17]; buffer[0][2] =   data[18]; buffer[0][3] =   data[36]; buffer[0][4] =   data[37]; buffer[0][5] =   data[38]; buffer[0][6] =   data[56]; buffer[0][7] =   data[57]; buffer[0][8] =   data[58];

        }
        if (partition ==  38) {
            buffer[0][0] =   data[17]; buffer[0][1] =   data[18]; buffer[0][2] =   data[19]; buffer[0][3] =   data[37]; buffer[0][4] =   data[38]; buffer[0][5] =   data[39]; buffer[0][6] =   data[57]; buffer[0][7] =   data[58]; buffer[0][8] =   data[59];

        }
        if (partition ==  39) {
            buffer[0][0] =   data[18]; buffer[0][1] =   data[19]; buffer[0][2] =          0; buffer[0][3] =   data[38]; buffer[0][4] =   data[39]; buffer[0][5] =          0; buffer[0][6] =   data[58]; buffer[0][7] =   data[59]; buffer[0][8] =          0;

        }
        if (partition ==  40) {
            buffer[0][0] =          0; buffer[0][1] =   data[20]; buffer[0][2] =   data[21]; buffer[0][3] =          0; buffer[0][4] =   data[40]; buffer[0][5] =   data[41]; buffer[0][6] =          0; buffer[0][7] =   data[60]; buffer[0][8] =   data[61];

        }
        if (partition ==  41) {
            buffer[0][0] =   data[20]; buffer[0][1] =   data[21]; buffer[0][2] =   data[22]; buffer[0][3] =   data[40]; buffer[0][4] =   data[41]; buffer[0][5] =   data[42]; buffer[0][6] =   data[60]; buffer[0][7] =   data[61]; buffer[0][8] =   data[62];

        }
        if (partition ==  42) {
            buffer[0][0] =   data[21]; buffer[0][1] =   data[22]; buffer[0][2] =   data[23]; buffer[0][3] =   data[41]; buffer[0][4] =   data[42]; buffer[0][5] =   data[43]; buffer[0][6] =   data[61]; buffer[0][7] =   data[62]; buffer[0][8] =   data[63];

        }
        if (partition ==  43) {
            buffer[0][0] =   data[22]; buffer[0][1] =   data[23]; buffer[0][2] =   data[24]; buffer[0][3] =   data[42]; buffer[0][4] =   data[43]; buffer[0][5] =   data[44]; buffer[0][6] =   data[62]; buffer[0][7] =   data[63]; buffer[0][8] =   data[64];

        }
        if (partition ==  44) {
            buffer[0][0] =   data[23]; buffer[0][1] =   data[24]; buffer[0][2] =   data[25]; buffer[0][3] =   data[43]; buffer[0][4] =   data[44]; buffer[0][5] =   data[45]; buffer[0][6] =   data[63]; buffer[0][7] =   data[64]; buffer[0][8] =   data[65];

        }
        if (partition ==  45) {
            buffer[0][0] =   data[24]; buffer[0][1] =   data[25]; buffer[0][2] =   data[26]; buffer[0][3] =   data[44]; buffer[0][4] =   data[45]; buffer[0][5] =   data[46]; buffer[0][6] =   data[64]; buffer[0][7] =   data[65]; buffer[0][8] =   data[66];

        }
        if (partition ==  46) {
            buffer[0][0] =   data[25]; buffer[0][1] =   data[26]; buffer[0][2] =   data[27]; buffer[0][3] =   data[45]; buffer[0][4] =   data[46]; buffer[0][5] =   data[47]; buffer[0][6] =   data[65]; buffer[0][7] =   data[66]; buffer[0][8] =   data[67];

        }
        if (partition ==  47) {
            buffer[0][0] =   data[26]; buffer[0][1] =   data[27]; buffer[0][2] =   data[28]; buffer[0][3] =   data[46]; buffer[0][4] =   data[47]; buffer[0][5] =   data[48]; buffer[0][6] =   data[66]; buffer[0][7] =   data[67]; buffer[0][8] =   data[68];

        }
        if (partition ==  48) {
            buffer[0][0] =   data[27]; buffer[0][1] =   data[28]; buffer[0][2] =   data[29]; buffer[0][3] =   data[47]; buffer[0][4] =   data[48]; buffer[0][5] =   data[49]; buffer[0][6] =   data[67]; buffer[0][7] =   data[68]; buffer[0][8] =   data[69];

        }
        if (partition ==  49) {
            buffer[0][0] =   data[28]; buffer[0][1] =   data[29]; buffer[0][2] =   data[30]; buffer[0][3] =   data[48]; buffer[0][4] =   data[49]; buffer[0][5] =   data[50]; buffer[0][6] =   data[68]; buffer[0][7] =   data[69]; buffer[0][8] =   data[70];

        }
        if (partition ==  50) {
            buffer[0][0] =   data[29]; buffer[0][1] =   data[30]; buffer[0][2] =   data[31]; buffer[0][3] =   data[49]; buffer[0][4] =   data[50]; buffer[0][5] =   data[51]; buffer[0][6] =   data[69]; buffer[0][7] =   data[70]; buffer[0][8] =   data[71];

        }
        if (partition ==  51) {
            buffer[0][0] =   data[30]; buffer[0][1] =   data[31]; buffer[0][2] =   data[32]; buffer[0][3] =   data[50]; buffer[0][4] =   data[51]; buffer[0][5] =   data[52]; buffer[0][6] =   data[70]; buffer[0][7] =   data[71]; buffer[0][8] =   data[72];

        }
        if (partition ==  52) {
            buffer[0][0] =   data[31]; buffer[0][1] =   data[32]; buffer[0][2] =   data[33]; buffer[0][3] =   data[51]; buffer[0][4] =   data[52]; buffer[0][5] =   data[53]; buffer[0][6] =   data[71]; buffer[0][7] =   data[72]; buffer[0][8] =   data[73];

        }
        if (partition ==  53) {
            buffer[0][0] =   data[32]; buffer[0][1] =   data[33]; buffer[0][2] =   data[34]; buffer[0][3] =   data[52]; buffer[0][4] =   data[53]; buffer[0][5] =   data[54]; buffer[0][6] =   data[72]; buffer[0][7] =   data[73]; buffer[0][8] =   data[74];

        }
        if (partition ==  54) {
            buffer[0][0] =   data[33]; buffer[0][1] =   data[34]; buffer[0][2] =   data[35]; buffer[0][3] =   data[53]; buffer[0][4] =   data[54]; buffer[0][5] =   data[55]; buffer[0][6] =   data[73]; buffer[0][7] =   data[74]; buffer[0][8] =   data[75];

        }
        if (partition ==  55) {
            buffer[0][0] =   data[34]; buffer[0][1] =   data[35]; buffer[0][2] =   data[36]; buffer[0][3] =   data[54]; buffer[0][4] =   data[55]; buffer[0][5] =   data[56]; buffer[0][6] =   data[74]; buffer[0][7] =   data[75]; buffer[0][8] =   data[76];

        }
        if (partition ==  56) {
            buffer[0][0] =   data[35]; buffer[0][1] =   data[36]; buffer[0][2] =   data[37]; buffer[0][3] =   data[55]; buffer[0][4] =   data[56]; buffer[0][5] =   data[57]; buffer[0][6] =   data[75]; buffer[0][7] =   data[76]; buffer[0][8] =   data[77];

        }
        if (partition ==  57) {
            buffer[0][0] =   data[36]; buffer[0][1] =   data[37]; buffer[0][2] =   data[38]; buffer[0][3] =   data[56]; buffer[0][4] =   data[57]; buffer[0][5] =   data[58]; buffer[0][6] =   data[76]; buffer[0][7] =   data[77]; buffer[0][8] =   data[78];

        }
        if (partition ==  58) {
            buffer[0][0] =   data[37]; buffer[0][1] =   data[38]; buffer[0][2] =   data[39]; buffer[0][3] =   data[57]; buffer[0][4] =   data[58]; buffer[0][5] =   data[59]; buffer[0][6] =   data[77]; buffer[0][7] =   data[78]; buffer[0][8] =   data[79];

        }
        if (partition ==  59) {
            buffer[0][0] =   data[38]; buffer[0][1] =   data[39]; buffer[0][2] =          0; buffer[0][3] =   data[58]; buffer[0][4] =   data[59]; buffer[0][5] =          0; buffer[0][6] =   data[78]; buffer[0][7] =   data[79]; buffer[0][8] =          0;

        }
        if (partition ==  60) {
            buffer[0][0] =          0; buffer[0][1] =   data[40]; buffer[0][2] =   data[41]; buffer[0][3] =          0; buffer[0][4] =   data[60]; buffer[0][5] =   data[61]; buffer[0][6] =          0; buffer[0][7] =   data[80]; buffer[0][8] =   data[81];

        }
        if (partition ==  61) {
            buffer[0][0] =   data[40]; buffer[0][1] =   data[41]; buffer[0][2] =   data[42]; buffer[0][3] =   data[60]; buffer[0][4] =   data[61]; buffer[0][5] =   data[62]; buffer[0][6] =   data[80]; buffer[0][7] =   data[81]; buffer[0][8] =   data[82];

        }
        if (partition ==  62) {
            buffer[0][0] =   data[41]; buffer[0][1] =   data[42]; buffer[0][2] =   data[43]; buffer[0][3] =   data[61]; buffer[0][4] =   data[62]; buffer[0][5] =   data[63]; buffer[0][6] =   data[81]; buffer[0][7] =   data[82]; buffer[0][8] =   data[83];

        }
        if (partition ==  63) {
            buffer[0][0] =   data[42]; buffer[0][1] =   data[43]; buffer[0][2] =   data[44]; buffer[0][3] =   data[62]; buffer[0][4] =   data[63]; buffer[0][5] =   data[64]; buffer[0][6] =   data[82]; buffer[0][7] =   data[83]; buffer[0][8] =   data[84];

        }
        if (partition ==  64) {
            buffer[0][0] =   data[43]; buffer[0][1] =   data[44]; buffer[0][2] =   data[45]; buffer[0][3] =   data[63]; buffer[0][4] =   data[64]; buffer[0][5] =   data[65]; buffer[0][6] =   data[83]; buffer[0][7] =   data[84]; buffer[0][8] =   data[85];

        }
        if (partition ==  65) {
            buffer[0][0] =   data[44]; buffer[0][1] =   data[45]; buffer[0][2] =   data[46]; buffer[0][3] =   data[64]; buffer[0][4] =   data[65]; buffer[0][5] =   data[66]; buffer[0][6] =   data[84]; buffer[0][7] =   data[85]; buffer[0][8] =   data[86];

        }
        if (partition ==  66) {
            buffer[0][0] =   data[45]; buffer[0][1] =   data[46]; buffer[0][2] =   data[47]; buffer[0][3] =   data[65]; buffer[0][4] =   data[66]; buffer[0][5] =   data[67]; buffer[0][6] =   data[85]; buffer[0][7] =   data[86]; buffer[0][8] =   data[87];

        }
        if (partition ==  67) {
            buffer[0][0] =   data[46]; buffer[0][1] =   data[47]; buffer[0][2] =   data[48]; buffer[0][3] =   data[66]; buffer[0][4] =   data[67]; buffer[0][5] =   data[68]; buffer[0][6] =   data[86]; buffer[0][7] =   data[87]; buffer[0][8] =   data[88];

        }
        if (partition ==  68) {
            buffer[0][0] =   data[47]; buffer[0][1] =   data[48]; buffer[0][2] =   data[49]; buffer[0][3] =   data[67]; buffer[0][4] =   data[68]; buffer[0][5] =   data[69]; buffer[0][6] =   data[87]; buffer[0][7] =   data[88]; buffer[0][8] =   data[89];

        }
        if (partition ==  69) {
            buffer[0][0] =   data[48]; buffer[0][1] =   data[49]; buffer[0][2] =   data[50]; buffer[0][3] =   data[68]; buffer[0][4] =   data[69]; buffer[0][5] =   data[70]; buffer[0][6] =   data[88]; buffer[0][7] =   data[89]; buffer[0][8] =   data[90];

        }
        if (partition ==  70) {
            buffer[0][0] =   data[49]; buffer[0][1] =   data[50]; buffer[0][2] =   data[51]; buffer[0][3] =   data[69]; buffer[0][4] =   data[70]; buffer[0][5] =   data[71]; buffer[0][6] =   data[89]; buffer[0][7] =   data[90]; buffer[0][8] =   data[91];

        }
        if (partition ==  71) {
            buffer[0][0] =   data[50]; buffer[0][1] =   data[51]; buffer[0][2] =   data[52]; buffer[0][3] =   data[70]; buffer[0][4] =   data[71]; buffer[0][5] =   data[72]; buffer[0][6] =   data[90]; buffer[0][7] =   data[91]; buffer[0][8] =   data[92];

        }
        if (partition ==  72) {
            buffer[0][0] =   data[51]; buffer[0][1] =   data[52]; buffer[0][2] =   data[53]; buffer[0][3] =   data[71]; buffer[0][4] =   data[72]; buffer[0][5] =   data[73]; buffer[0][6] =   data[91]; buffer[0][7] =   data[92]; buffer[0][8] =   data[93];

        }
        if (partition ==  73) {
            buffer[0][0] =   data[52]; buffer[0][1] =   data[53]; buffer[0][2] =   data[54]; buffer[0][3] =   data[72]; buffer[0][4] =   data[73]; buffer[0][5] =   data[74]; buffer[0][6] =   data[92]; buffer[0][7] =   data[93]; buffer[0][8] =   data[94];

        }
        if (partition ==  74) {
            buffer[0][0] =   data[53]; buffer[0][1] =   data[54]; buffer[0][2] =   data[55]; buffer[0][3] =   data[73]; buffer[0][4] =   data[74]; buffer[0][5] =   data[75]; buffer[0][6] =   data[93]; buffer[0][7] =   data[94]; buffer[0][8] =   data[95];

        }
        if (partition ==  75) {
            buffer[0][0] =   data[54]; buffer[0][1] =   data[55]; buffer[0][2] =   data[56]; buffer[0][3] =   data[74]; buffer[0][4] =   data[75]; buffer[0][5] =   data[76]; buffer[0][6] =   data[94]; buffer[0][7] =   data[95]; buffer[0][8] =   data[96];

        }
        if (partition ==  76) {
            buffer[0][0] =   data[55]; buffer[0][1] =   data[56]; buffer[0][2] =   data[57]; buffer[0][3] =   data[75]; buffer[0][4] =   data[76]; buffer[0][5] =   data[77]; buffer[0][6] =   data[95]; buffer[0][7] =   data[96]; buffer[0][8] =   data[97];

        }
        if (partition ==  77) {
            buffer[0][0] =   data[56]; buffer[0][1] =   data[57]; buffer[0][2] =   data[58]; buffer[0][3] =   data[76]; buffer[0][4] =   data[77]; buffer[0][5] =   data[78]; buffer[0][6] =   data[96]; buffer[0][7] =   data[97]; buffer[0][8] =   data[98];

        }
        if (partition ==  78) {
            buffer[0][0] =   data[57]; buffer[0][1] =   data[58]; buffer[0][2] =   data[59]; buffer[0][3] =   data[77]; buffer[0][4] =   data[78]; buffer[0][5] =   data[79]; buffer[0][6] =   data[97]; buffer[0][7] =   data[98]; buffer[0][8] =   data[99];

        }
        if (partition ==  79) {
            buffer[0][0] =   data[58]; buffer[0][1] =   data[59]; buffer[0][2] =          0; buffer[0][3] =   data[78]; buffer[0][4] =   data[79]; buffer[0][5] =          0; buffer[0][6] =   data[98]; buffer[0][7] =   data[99]; buffer[0][8] =          0;

        }
        if (partition ==  80) {
            buffer[0][0] =          0; buffer[0][1] =   data[60]; buffer[0][2] =   data[61]; buffer[0][3] =          0; buffer[0][4] =   data[80]; buffer[0][5] =   data[81]; buffer[0][6] =          0; buffer[0][7] =  data[100]; buffer[0][8] =  data[101];

        }
        if (partition ==  81) {
            buffer[0][0] =   data[60]; buffer[0][1] =   data[61]; buffer[0][2] =   data[62]; buffer[0][3] =   data[80]; buffer[0][4] =   data[81]; buffer[0][5] =   data[82]; buffer[0][6] =  data[100]; buffer[0][7] =  data[101]; buffer[0][8] =  data[102];

        }
        if (partition ==  82) {
            buffer[0][0] =   data[61]; buffer[0][1] =   data[62]; buffer[0][2] =   data[63]; buffer[0][3] =   data[81]; buffer[0][4] =   data[82]; buffer[0][5] =   data[83]; buffer[0][6] =  data[101]; buffer[0][7] =  data[102]; buffer[0][8] =  data[103];

        }
        if (partition ==  83) {
            buffer[0][0] =   data[62]; buffer[0][1] =   data[63]; buffer[0][2] =   data[64]; buffer[0][3] =   data[82]; buffer[0][4] =   data[83]; buffer[0][5] =   data[84]; buffer[0][6] =  data[102]; buffer[0][7] =  data[103]; buffer[0][8] =  data[104];

        }
        if (partition ==  84) {
            buffer[0][0] =   data[63]; buffer[0][1] =   data[64]; buffer[0][2] =   data[65]; buffer[0][3] =   data[83]; buffer[0][4] =   data[84]; buffer[0][5] =   data[85]; buffer[0][6] =  data[103]; buffer[0][7] =  data[104]; buffer[0][8] =  data[105];

        }
        if (partition ==  85) {
            buffer[0][0] =   data[64]; buffer[0][1] =   data[65]; buffer[0][2] =   data[66]; buffer[0][3] =   data[84]; buffer[0][4] =   data[85]; buffer[0][5] =   data[86]; buffer[0][6] =  data[104]; buffer[0][7] =  data[105]; buffer[0][8] =  data[106];

        }
        if (partition ==  86) {
            buffer[0][0] =   data[65]; buffer[0][1] =   data[66]; buffer[0][2] =   data[67]; buffer[0][3] =   data[85]; buffer[0][4] =   data[86]; buffer[0][5] =   data[87]; buffer[0][6] =  data[105]; buffer[0][7] =  data[106]; buffer[0][8] =  data[107];

        }
        if (partition ==  87) {
            buffer[0][0] =   data[66]; buffer[0][1] =   data[67]; buffer[0][2] =   data[68]; buffer[0][3] =   data[86]; buffer[0][4] =   data[87]; buffer[0][5] =   data[88]; buffer[0][6] =  data[106]; buffer[0][7] =  data[107]; buffer[0][8] =  data[108];

        }
        if (partition ==  88) {
            buffer[0][0] =   data[67]; buffer[0][1] =   data[68]; buffer[0][2] =   data[69]; buffer[0][3] =   data[87]; buffer[0][4] =   data[88]; buffer[0][5] =   data[89]; buffer[0][6] =  data[107]; buffer[0][7] =  data[108]; buffer[0][8] =  data[109];

        }
        if (partition ==  89) {
            buffer[0][0] =   data[68]; buffer[0][1] =   data[69]; buffer[0][2] =   data[70]; buffer[0][3] =   data[88]; buffer[0][4] =   data[89]; buffer[0][5] =   data[90]; buffer[0][6] =  data[108]; buffer[0][7] =  data[109]; buffer[0][8] =  data[110];

        }
        if (partition ==  90) {
            buffer[0][0] =   data[69]; buffer[0][1] =   data[70]; buffer[0][2] =   data[71]; buffer[0][3] =   data[89]; buffer[0][4] =   data[90]; buffer[0][5] =   data[91]; buffer[0][6] =  data[109]; buffer[0][7] =  data[110]; buffer[0][8] =  data[111];

        }
        if (partition ==  91) {
            buffer[0][0] =   data[70]; buffer[0][1] =   data[71]; buffer[0][2] =   data[72]; buffer[0][3] =   data[90]; buffer[0][4] =   data[91]; buffer[0][5] =   data[92]; buffer[0][6] =  data[110]; buffer[0][7] =  data[111]; buffer[0][8] =  data[112];

        }
        if (partition ==  92) {
            buffer[0][0] =   data[71]; buffer[0][1] =   data[72]; buffer[0][2] =   data[73]; buffer[0][3] =   data[91]; buffer[0][4] =   data[92]; buffer[0][5] =   data[93]; buffer[0][6] =  data[111]; buffer[0][7] =  data[112]; buffer[0][8] =  data[113];

        }
        if (partition ==  93) {
            buffer[0][0] =   data[72]; buffer[0][1] =   data[73]; buffer[0][2] =   data[74]; buffer[0][3] =   data[92]; buffer[0][4] =   data[93]; buffer[0][5] =   data[94]; buffer[0][6] =  data[112]; buffer[0][7] =  data[113]; buffer[0][8] =  data[114];

        }
        if (partition ==  94) {
            buffer[0][0] =   data[73]; buffer[0][1] =   data[74]; buffer[0][2] =   data[75]; buffer[0][3] =   data[93]; buffer[0][4] =   data[94]; buffer[0][5] =   data[95]; buffer[0][6] =  data[113]; buffer[0][7] =  data[114]; buffer[0][8] =  data[115];

        }
        if (partition ==  95) {
            buffer[0][0] =   data[74]; buffer[0][1] =   data[75]; buffer[0][2] =   data[76]; buffer[0][3] =   data[94]; buffer[0][4] =   data[95]; buffer[0][5] =   data[96]; buffer[0][6] =  data[114]; buffer[0][7] =  data[115]; buffer[0][8] =  data[116];

        }
        if (partition ==  96) {
            buffer[0][0] =   data[75]; buffer[0][1] =   data[76]; buffer[0][2] =   data[77]; buffer[0][3] =   data[95]; buffer[0][4] =   data[96]; buffer[0][5] =   data[97]; buffer[0][6] =  data[115]; buffer[0][7] =  data[116]; buffer[0][8] =  data[117];

        }
        if (partition ==  97) {
            buffer[0][0] =   data[76]; buffer[0][1] =   data[77]; buffer[0][2] =   data[78]; buffer[0][3] =   data[96]; buffer[0][4] =   data[97]; buffer[0][5] =   data[98]; buffer[0][6] =  data[116]; buffer[0][7] =  data[117]; buffer[0][8] =  data[118];

        }
        if (partition ==  98) {
            buffer[0][0] =   data[77]; buffer[0][1] =   data[78]; buffer[0][2] =   data[79]; buffer[0][3] =   data[97]; buffer[0][4] =   data[98]; buffer[0][5] =   data[99]; buffer[0][6] =  data[117]; buffer[0][7] =  data[118]; buffer[0][8] =  data[119];

        }
        if (partition ==  99) {
            buffer[0][0] =   data[78]; buffer[0][1] =   data[79]; buffer[0][2] =          0; buffer[0][3] =   data[98]; buffer[0][4] =   data[99]; buffer[0][5] =          0; buffer[0][6] =  data[118]; buffer[0][7] =  data[119]; buffer[0][8] =          0;

        }
        if (partition == 100) {
            buffer[0][0] =          0; buffer[0][1] =   data[80]; buffer[0][2] =   data[81]; buffer[0][3] =          0; buffer[0][4] =  data[100]; buffer[0][5] =  data[101]; buffer[0][6] =          0; buffer[0][7] =  data[120]; buffer[0][8] =  data[121];

        }
        if (partition == 101) {
            buffer[0][0] =   data[80]; buffer[0][1] =   data[81]; buffer[0][2] =   data[82]; buffer[0][3] =  data[100]; buffer[0][4] =  data[101]; buffer[0][5] =  data[102]; buffer[0][6] =  data[120]; buffer[0][7] =  data[121]; buffer[0][8] =  data[122];

        }
        if (partition == 102) {
            buffer[0][0] =   data[81]; buffer[0][1] =   data[82]; buffer[0][2] =   data[83]; buffer[0][3] =  data[101]; buffer[0][4] =  data[102]; buffer[0][5] =  data[103]; buffer[0][6] =  data[121]; buffer[0][7] =  data[122]; buffer[0][8] =  data[123];

        }
        if (partition == 103) {
            buffer[0][0] =   data[82]; buffer[0][1] =   data[83]; buffer[0][2] =   data[84]; buffer[0][3] =  data[102]; buffer[0][4] =  data[103]; buffer[0][5] =  data[104]; buffer[0][6] =  data[122]; buffer[0][7] =  data[123]; buffer[0][8] =  data[124];

        }
        if (partition == 104) {
            buffer[0][0] =   data[83]; buffer[0][1] =   data[84]; buffer[0][2] =   data[85]; buffer[0][3] =  data[103]; buffer[0][4] =  data[104]; buffer[0][5] =  data[105]; buffer[0][6] =  data[123]; buffer[0][7] =  data[124]; buffer[0][8] =  data[125];

        }
        if (partition == 105) {
            buffer[0][0] =   data[84]; buffer[0][1] =   data[85]; buffer[0][2] =   data[86]; buffer[0][3] =  data[104]; buffer[0][4] =  data[105]; buffer[0][5] =  data[106]; buffer[0][6] =  data[124]; buffer[0][7] =  data[125]; buffer[0][8] =  data[126];

        }
        if (partition == 106) {
            buffer[0][0] =   data[85]; buffer[0][1] =   data[86]; buffer[0][2] =   data[87]; buffer[0][3] =  data[105]; buffer[0][4] =  data[106]; buffer[0][5] =  data[107]; buffer[0][6] =  data[125]; buffer[0][7] =  data[126]; buffer[0][8] =  data[127];

        }
        if (partition == 107) {
            buffer[0][0] =   data[86]; buffer[0][1] =   data[87]; buffer[0][2] =   data[88]; buffer[0][3] =  data[106]; buffer[0][4] =  data[107]; buffer[0][5] =  data[108]; buffer[0][6] =  data[126]; buffer[0][7] =  data[127]; buffer[0][8] =  data[128];

        }
        if (partition == 108) {
            buffer[0][0] =   data[87]; buffer[0][1] =   data[88]; buffer[0][2] =   data[89]; buffer[0][3] =  data[107]; buffer[0][4] =  data[108]; buffer[0][5] =  data[109]; buffer[0][6] =  data[127]; buffer[0][7] =  data[128]; buffer[0][8] =  data[129];

        }
        if (partition == 109) {
            buffer[0][0] =   data[88]; buffer[0][1] =   data[89]; buffer[0][2] =   data[90]; buffer[0][3] =  data[108]; buffer[0][4] =  data[109]; buffer[0][5] =  data[110]; buffer[0][6] =  data[128]; buffer[0][7] =  data[129]; buffer[0][8] =  data[130];

        }
        if (partition == 110) {
            buffer[0][0] =   data[89]; buffer[0][1] =   data[90]; buffer[0][2] =   data[91]; buffer[0][3] =  data[109]; buffer[0][4] =  data[110]; buffer[0][5] =  data[111]; buffer[0][6] =  data[129]; buffer[0][7] =  data[130]; buffer[0][8] =  data[131];

        }
        if (partition == 111) {
            buffer[0][0] =   data[90]; buffer[0][1] =   data[91]; buffer[0][2] =   data[92]; buffer[0][3] =  data[110]; buffer[0][4] =  data[111]; buffer[0][5] =  data[112]; buffer[0][6] =  data[130]; buffer[0][7] =  data[131]; buffer[0][8] =  data[132];

        }
        if (partition == 112) {
            buffer[0][0] =   data[91]; buffer[0][1] =   data[92]; buffer[0][2] =   data[93]; buffer[0][3] =  data[111]; buffer[0][4] =  data[112]; buffer[0][5] =  data[113]; buffer[0][6] =  data[131]; buffer[0][7] =  data[132]; buffer[0][8] =  data[133];

        }
        if (partition == 113) {
            buffer[0][0] =   data[92]; buffer[0][1] =   data[93]; buffer[0][2] =   data[94]; buffer[0][3] =  data[112]; buffer[0][4] =  data[113]; buffer[0][5] =  data[114]; buffer[0][6] =  data[132]; buffer[0][7] =  data[133]; buffer[0][8] =  data[134];

        }
        if (partition == 114) {
            buffer[0][0] =   data[93]; buffer[0][1] =   data[94]; buffer[0][2] =   data[95]; buffer[0][3] =  data[113]; buffer[0][4] =  data[114]; buffer[0][5] =  data[115]; buffer[0][6] =  data[133]; buffer[0][7] =  data[134]; buffer[0][8] =  data[135];

        }
        if (partition == 115) {
            buffer[0][0] =   data[94]; buffer[0][1] =   data[95]; buffer[0][2] =   data[96]; buffer[0][3] =  data[114]; buffer[0][4] =  data[115]; buffer[0][5] =  data[116]; buffer[0][6] =  data[134]; buffer[0][7] =  data[135]; buffer[0][8] =  data[136];

        }
        if (partition == 116) {
            buffer[0][0] =   data[95]; buffer[0][1] =   data[96]; buffer[0][2] =   data[97]; buffer[0][3] =  data[115]; buffer[0][4] =  data[116]; buffer[0][5] =  data[117]; buffer[0][6] =  data[135]; buffer[0][7] =  data[136]; buffer[0][8] =  data[137];

        }
        if (partition == 117) {
            buffer[0][0] =   data[96]; buffer[0][1] =   data[97]; buffer[0][2] =   data[98]; buffer[0][3] =  data[116]; buffer[0][4] =  data[117]; buffer[0][5] =  data[118]; buffer[0][6] =  data[136]; buffer[0][7] =  data[137]; buffer[0][8] =  data[138];

        }
        if (partition == 118) {
            buffer[0][0] =   data[97]; buffer[0][1] =   data[98]; buffer[0][2] =   data[99]; buffer[0][3] =  data[117]; buffer[0][4] =  data[118]; buffer[0][5] =  data[119]; buffer[0][6] =  data[137]; buffer[0][7] =  data[138]; buffer[0][8] =  data[139];

        }
        if (partition == 119) {
            buffer[0][0] =   data[98]; buffer[0][1] =   data[99]; buffer[0][2] =          0; buffer[0][3] =  data[118]; buffer[0][4] =  data[119]; buffer[0][5] =          0; buffer[0][6] =  data[138]; buffer[0][7] =  data[139]; buffer[0][8] =          0;

        }
        if (partition == 120) {
            buffer[0][0] =          0; buffer[0][1] =  data[100]; buffer[0][2] =  data[101]; buffer[0][3] =          0; buffer[0][4] =  data[120]; buffer[0][5] =  data[121]; buffer[0][6] =          0; buffer[0][7] =  data[140]; buffer[0][8] =  data[141];

        }
        if (partition == 121) {
            buffer[0][0] =  data[100]; buffer[0][1] =  data[101]; buffer[0][2] =  data[102]; buffer[0][3] =  data[120]; buffer[0][4] =  data[121]; buffer[0][5] =  data[122]; buffer[0][6] =  data[140]; buffer[0][7] =  data[141]; buffer[0][8] =  data[142];

        }
        if (partition == 122) {
            buffer[0][0] =  data[101]; buffer[0][1] =  data[102]; buffer[0][2] =  data[103]; buffer[0][3] =  data[121]; buffer[0][4] =  data[122]; buffer[0][5] =  data[123]; buffer[0][6] =  data[141]; buffer[0][7] =  data[142]; buffer[0][8] =  data[143];

        }
        if (partition == 123) {
            buffer[0][0] =  data[102]; buffer[0][1] =  data[103]; buffer[0][2] =  data[104]; buffer[0][3] =  data[122]; buffer[0][4] =  data[123]; buffer[0][5] =  data[124]; buffer[0][6] =  data[142]; buffer[0][7] =  data[143]; buffer[0][8] =  data[144];

        }
        if (partition == 124) {
            buffer[0][0] =  data[103]; buffer[0][1] =  data[104]; buffer[0][2] =  data[105]; buffer[0][3] =  data[123]; buffer[0][4] =  data[124]; buffer[0][5] =  data[125]; buffer[0][6] =  data[143]; buffer[0][7] =  data[144]; buffer[0][8] =  data[145];

        }
        if (partition == 125) {
            buffer[0][0] =  data[104]; buffer[0][1] =  data[105]; buffer[0][2] =  data[106]; buffer[0][3] =  data[124]; buffer[0][4] =  data[125]; buffer[0][5] =  data[126]; buffer[0][6] =  data[144]; buffer[0][7] =  data[145]; buffer[0][8] =  data[146];

        }
        if (partition == 126) {
            buffer[0][0] =  data[105]; buffer[0][1] =  data[106]; buffer[0][2] =  data[107]; buffer[0][3] =  data[125]; buffer[0][4] =  data[126]; buffer[0][5] =  data[127]; buffer[0][6] =  data[145]; buffer[0][7] =  data[146]; buffer[0][8] =  data[147];

        }
        if (partition == 127) {
            buffer[0][0] =  data[106]; buffer[0][1] =  data[107]; buffer[0][2] =  data[108]; buffer[0][3] =  data[126]; buffer[0][4] =  data[127]; buffer[0][5] =  data[128]; buffer[0][6] =  data[146]; buffer[0][7] =  data[147]; buffer[0][8] =  data[148];

        }
        if (partition == 128) {
            buffer[0][0] =  data[107]; buffer[0][1] =  data[108]; buffer[0][2] =  data[109]; buffer[0][3] =  data[127]; buffer[0][4] =  data[128]; buffer[0][5] =  data[129]; buffer[0][6] =  data[147]; buffer[0][7] =  data[148]; buffer[0][8] =  data[149];

        }
        if (partition == 129) {
            buffer[0][0] =  data[108]; buffer[0][1] =  data[109]; buffer[0][2] =  data[110]; buffer[0][3] =  data[128]; buffer[0][4] =  data[129]; buffer[0][5] =  data[130]; buffer[0][6] =  data[148]; buffer[0][7] =  data[149]; buffer[0][8] =  data[150];

        }
        if (partition == 130) {
            buffer[0][0] =  data[109]; buffer[0][1] =  data[110]; buffer[0][2] =  data[111]; buffer[0][3] =  data[129]; buffer[0][4] =  data[130]; buffer[0][5] =  data[131]; buffer[0][6] =  data[149]; buffer[0][7] =  data[150]; buffer[0][8] =  data[151];

        }
        if (partition == 131) {
            buffer[0][0] =  data[110]; buffer[0][1] =  data[111]; buffer[0][2] =  data[112]; buffer[0][3] =  data[130]; buffer[0][4] =  data[131]; buffer[0][5] =  data[132]; buffer[0][6] =  data[150]; buffer[0][7] =  data[151]; buffer[0][8] =  data[152];

        }
        if (partition == 132) {
            buffer[0][0] =  data[111]; buffer[0][1] =  data[112]; buffer[0][2] =  data[113]; buffer[0][3] =  data[131]; buffer[0][4] =  data[132]; buffer[0][5] =  data[133]; buffer[0][6] =  data[151]; buffer[0][7] =  data[152]; buffer[0][8] =  data[153];

        }
        if (partition == 133) {
            buffer[0][0] =  data[112]; buffer[0][1] =  data[113]; buffer[0][2] =  data[114]; buffer[0][3] =  data[132]; buffer[0][4] =  data[133]; buffer[0][5] =  data[134]; buffer[0][6] =  data[152]; buffer[0][7] =  data[153]; buffer[0][8] =  data[154];

        }
        if (partition == 134) {
            buffer[0][0] =  data[113]; buffer[0][1] =  data[114]; buffer[0][2] =  data[115]; buffer[0][3] =  data[133]; buffer[0][4] =  data[134]; buffer[0][5] =  data[135]; buffer[0][6] =  data[153]; buffer[0][7] =  data[154]; buffer[0][8] =  data[155];

        }
        if (partition == 135) {
            buffer[0][0] =  data[114]; buffer[0][1] =  data[115]; buffer[0][2] =  data[116]; buffer[0][3] =  data[134]; buffer[0][4] =  data[135]; buffer[0][5] =  data[136]; buffer[0][6] =  data[154]; buffer[0][7] =  data[155]; buffer[0][8] =  data[156];

        }
        if (partition == 136) {
            buffer[0][0] =  data[115]; buffer[0][1] =  data[116]; buffer[0][2] =  data[117]; buffer[0][3] =  data[135]; buffer[0][4] =  data[136]; buffer[0][5] =  data[137]; buffer[0][6] =  data[155]; buffer[0][7] =  data[156]; buffer[0][8] =  data[157];

        }
        if (partition == 137) {
            buffer[0][0] =  data[116]; buffer[0][1] =  data[117]; buffer[0][2] =  data[118]; buffer[0][3] =  data[136]; buffer[0][4] =  data[137]; buffer[0][5] =  data[138]; buffer[0][6] =  data[156]; buffer[0][7] =  data[157]; buffer[0][8] =  data[158];

        }
        if (partition == 138) {
            buffer[0][0] =  data[117]; buffer[0][1] =  data[118]; buffer[0][2] =  data[119]; buffer[0][3] =  data[137]; buffer[0][4] =  data[138]; buffer[0][5] =  data[139]; buffer[0][6] =  data[157]; buffer[0][7] =  data[158]; buffer[0][8] =  data[159];

        }
        if (partition == 139) {
            buffer[0][0] =  data[118]; buffer[0][1] =  data[119]; buffer[0][2] =          0; buffer[0][3] =  data[138]; buffer[0][4] =  data[139]; buffer[0][5] =          0; buffer[0][6] =  data[158]; buffer[0][7] =  data[159]; buffer[0][8] =          0;

        }
        if (partition == 140) {
            buffer[0][0] =          0; buffer[0][1] =  data[120]; buffer[0][2] =  data[121]; buffer[0][3] =          0; buffer[0][4] =  data[140]; buffer[0][5] =  data[141]; buffer[0][6] =          0; buffer[0][7] =  data[160]; buffer[0][8] =  data[161];

        }
        if (partition == 141) {
            buffer[0][0] =  data[120]; buffer[0][1] =  data[121]; buffer[0][2] =  data[122]; buffer[0][3] =  data[140]; buffer[0][4] =  data[141]; buffer[0][5] =  data[142]; buffer[0][6] =  data[160]; buffer[0][7] =  data[161]; buffer[0][8] =  data[162];

        }
        if (partition == 142) {
            buffer[0][0] =  data[121]; buffer[0][1] =  data[122]; buffer[0][2] =  data[123]; buffer[0][3] =  data[141]; buffer[0][4] =  data[142]; buffer[0][5] =  data[143]; buffer[0][6] =  data[161]; buffer[0][7] =  data[162]; buffer[0][8] =  data[163];

        }
        if (partition == 143) {
            buffer[0][0] =  data[122]; buffer[0][1] =  data[123]; buffer[0][2] =  data[124]; buffer[0][3] =  data[142]; buffer[0][4] =  data[143]; buffer[0][5] =  data[144]; buffer[0][6] =  data[162]; buffer[0][7] =  data[163]; buffer[0][8] =  data[164];

        }
        if (partition == 144) {
            buffer[0][0] =  data[123]; buffer[0][1] =  data[124]; buffer[0][2] =  data[125]; buffer[0][3] =  data[143]; buffer[0][4] =  data[144]; buffer[0][5] =  data[145]; buffer[0][6] =  data[163]; buffer[0][7] =  data[164]; buffer[0][8] =  data[165];

        }
        if (partition == 145) {
            buffer[0][0] =  data[124]; buffer[0][1] =  data[125]; buffer[0][2] =  data[126]; buffer[0][3] =  data[144]; buffer[0][4] =  data[145]; buffer[0][5] =  data[146]; buffer[0][6] =  data[164]; buffer[0][7] =  data[165]; buffer[0][8] =  data[166];

        }
        if (partition == 146) {
            buffer[0][0] =  data[125]; buffer[0][1] =  data[126]; buffer[0][2] =  data[127]; buffer[0][3] =  data[145]; buffer[0][4] =  data[146]; buffer[0][5] =  data[147]; buffer[0][6] =  data[165]; buffer[0][7] =  data[166]; buffer[0][8] =  data[167];

        }
        if (partition == 147) {
            buffer[0][0] =  data[126]; buffer[0][1] =  data[127]; buffer[0][2] =  data[128]; buffer[0][3] =  data[146]; buffer[0][4] =  data[147]; buffer[0][5] =  data[148]; buffer[0][6] =  data[166]; buffer[0][7] =  data[167]; buffer[0][8] =  data[168];

        }
        if (partition == 148) {
            buffer[0][0] =  data[127]; buffer[0][1] =  data[128]; buffer[0][2] =  data[129]; buffer[0][3] =  data[147]; buffer[0][4] =  data[148]; buffer[0][5] =  data[149]; buffer[0][6] =  data[167]; buffer[0][7] =  data[168]; buffer[0][8] =  data[169];

        }
        if (partition == 149) {
            buffer[0][0] =  data[128]; buffer[0][1] =  data[129]; buffer[0][2] =  data[130]; buffer[0][3] =  data[148]; buffer[0][4] =  data[149]; buffer[0][5] =  data[150]; buffer[0][6] =  data[168]; buffer[0][7] =  data[169]; buffer[0][8] =  data[170];

        }
        if (partition == 150) {
            buffer[0][0] =  data[129]; buffer[0][1] =  data[130]; buffer[0][2] =  data[131]; buffer[0][3] =  data[149]; buffer[0][4] =  data[150]; buffer[0][5] =  data[151]; buffer[0][6] =  data[169]; buffer[0][7] =  data[170]; buffer[0][8] =  data[171];

        }
        if (partition == 151) {
            buffer[0][0] =  data[130]; buffer[0][1] =  data[131]; buffer[0][2] =  data[132]; buffer[0][3] =  data[150]; buffer[0][4] =  data[151]; buffer[0][5] =  data[152]; buffer[0][6] =  data[170]; buffer[0][7] =  data[171]; buffer[0][8] =  data[172];

        }
        if (partition == 152) {
            buffer[0][0] =  data[131]; buffer[0][1] =  data[132]; buffer[0][2] =  data[133]; buffer[0][3] =  data[151]; buffer[0][4] =  data[152]; buffer[0][5] =  data[153]; buffer[0][6] =  data[171]; buffer[0][7] =  data[172]; buffer[0][8] =  data[173];

        }
        if (partition == 153) {
            buffer[0][0] =  data[132]; buffer[0][1] =  data[133]; buffer[0][2] =  data[134]; buffer[0][3] =  data[152]; buffer[0][4] =  data[153]; buffer[0][5] =  data[154]; buffer[0][6] =  data[172]; buffer[0][7] =  data[173]; buffer[0][8] =  data[174];

        }
        if (partition == 154) {
            buffer[0][0] =  data[133]; buffer[0][1] =  data[134]; buffer[0][2] =  data[135]; buffer[0][3] =  data[153]; buffer[0][4] =  data[154]; buffer[0][5] =  data[155]; buffer[0][6] =  data[173]; buffer[0][7] =  data[174]; buffer[0][8] =  data[175];

        }
        if (partition == 155) {
            buffer[0][0] =  data[134]; buffer[0][1] =  data[135]; buffer[0][2] =  data[136]; buffer[0][3] =  data[154]; buffer[0][4] =  data[155]; buffer[0][5] =  data[156]; buffer[0][6] =  data[174]; buffer[0][7] =  data[175]; buffer[0][8] =  data[176];

        }
        if (partition == 156) {
            buffer[0][0] =  data[135]; buffer[0][1] =  data[136]; buffer[0][2] =  data[137]; buffer[0][3] =  data[155]; buffer[0][4] =  data[156]; buffer[0][5] =  data[157]; buffer[0][6] =  data[175]; buffer[0][7] =  data[176]; buffer[0][8] =  data[177];

        }
        if (partition == 157) {
            buffer[0][0] =  data[136]; buffer[0][1] =  data[137]; buffer[0][2] =  data[138]; buffer[0][3] =  data[156]; buffer[0][4] =  data[157]; buffer[0][5] =  data[158]; buffer[0][6] =  data[176]; buffer[0][7] =  data[177]; buffer[0][8] =  data[178];

        }
        if (partition == 158) {
            buffer[0][0] =  data[137]; buffer[0][1] =  data[138]; buffer[0][2] =  data[139]; buffer[0][3] =  data[157]; buffer[0][4] =  data[158]; buffer[0][5] =  data[159]; buffer[0][6] =  data[177]; buffer[0][7] =  data[178]; buffer[0][8] =  data[179];

        }
        if (partition == 159) {
            buffer[0][0] =  data[138]; buffer[0][1] =  data[139]; buffer[0][2] =          0; buffer[0][3] =  data[158]; buffer[0][4] =  data[159]; buffer[0][5] =          0; buffer[0][6] =  data[178]; buffer[0][7] =  data[179]; buffer[0][8] =          0;

        }
        if (partition == 160) {
            buffer[0][0] =          0; buffer[0][1] =  data[140]; buffer[0][2] =  data[141]; buffer[0][3] =          0; buffer[0][4] =  data[160]; buffer[0][5] =  data[161]; buffer[0][6] =          0; buffer[0][7] =  data[180]; buffer[0][8] =  data[181];

        }
        if (partition == 161) {
            buffer[0][0] =  data[140]; buffer[0][1] =  data[141]; buffer[0][2] =  data[142]; buffer[0][3] =  data[160]; buffer[0][4] =  data[161]; buffer[0][5] =  data[162]; buffer[0][6] =  data[180]; buffer[0][7] =  data[181]; buffer[0][8] =  data[182];

        }
        if (partition == 162) {
            buffer[0][0] =  data[141]; buffer[0][1] =  data[142]; buffer[0][2] =  data[143]; buffer[0][3] =  data[161]; buffer[0][4] =  data[162]; buffer[0][5] =  data[163]; buffer[0][6] =  data[181]; buffer[0][7] =  data[182]; buffer[0][8] =  data[183];

        }
        if (partition == 163) {
            buffer[0][0] =  data[142]; buffer[0][1] =  data[143]; buffer[0][2] =  data[144]; buffer[0][3] =  data[162]; buffer[0][4] =  data[163]; buffer[0][5] =  data[164]; buffer[0][6] =  data[182]; buffer[0][7] =  data[183]; buffer[0][8] =  data[184];

        }
        if (partition == 164) {
            buffer[0][0] =  data[143]; buffer[0][1] =  data[144]; buffer[0][2] =  data[145]; buffer[0][3] =  data[163]; buffer[0][4] =  data[164]; buffer[0][5] =  data[165]; buffer[0][6] =  data[183]; buffer[0][7] =  data[184]; buffer[0][8] =  data[185];

        }
        if (partition == 165) {
            buffer[0][0] =  data[144]; buffer[0][1] =  data[145]; buffer[0][2] =  data[146]; buffer[0][3] =  data[164]; buffer[0][4] =  data[165]; buffer[0][5] =  data[166]; buffer[0][6] =  data[184]; buffer[0][7] =  data[185]; buffer[0][8] =  data[186];

        }
        if (partition == 166) {
            buffer[0][0] =  data[145]; buffer[0][1] =  data[146]; buffer[0][2] =  data[147]; buffer[0][3] =  data[165]; buffer[0][4] =  data[166]; buffer[0][5] =  data[167]; buffer[0][6] =  data[185]; buffer[0][7] =  data[186]; buffer[0][8] =  data[187];

        }
        if (partition == 167) {
            buffer[0][0] =  data[146]; buffer[0][1] =  data[147]; buffer[0][2] =  data[148]; buffer[0][3] =  data[166]; buffer[0][4] =  data[167]; buffer[0][5] =  data[168]; buffer[0][6] =  data[186]; buffer[0][7] =  data[187]; buffer[0][8] =  data[188];

        }
        if (partition == 168) {
            buffer[0][0] =  data[147]; buffer[0][1] =  data[148]; buffer[0][2] =  data[149]; buffer[0][3] =  data[167]; buffer[0][4] =  data[168]; buffer[0][5] =  data[169]; buffer[0][6] =  data[187]; buffer[0][7] =  data[188]; buffer[0][8] =  data[189];

        }
        if (partition == 169) {
            buffer[0][0] =  data[148]; buffer[0][1] =  data[149]; buffer[0][2] =  data[150]; buffer[0][3] =  data[168]; buffer[0][4] =  data[169]; buffer[0][5] =  data[170]; buffer[0][6] =  data[188]; buffer[0][7] =  data[189]; buffer[0][8] =  data[190];

        }
        if (partition == 170) {
            buffer[0][0] =  data[149]; buffer[0][1] =  data[150]; buffer[0][2] =  data[151]; buffer[0][3] =  data[169]; buffer[0][4] =  data[170]; buffer[0][5] =  data[171]; buffer[0][6] =  data[189]; buffer[0][7] =  data[190]; buffer[0][8] =  data[191];

        }
        if (partition == 171) {
            buffer[0][0] =  data[150]; buffer[0][1] =  data[151]; buffer[0][2] =  data[152]; buffer[0][3] =  data[170]; buffer[0][4] =  data[171]; buffer[0][5] =  data[172]; buffer[0][6] =  data[190]; buffer[0][7] =  data[191]; buffer[0][8] =  data[192];

        }
        if (partition == 172) {
            buffer[0][0] =  data[151]; buffer[0][1] =  data[152]; buffer[0][2] =  data[153]; buffer[0][3] =  data[171]; buffer[0][4] =  data[172]; buffer[0][5] =  data[173]; buffer[0][6] =  data[191]; buffer[0][7] =  data[192]; buffer[0][8] =  data[193];

        }
        if (partition == 173) {
            buffer[0][0] =  data[152]; buffer[0][1] =  data[153]; buffer[0][2] =  data[154]; buffer[0][3] =  data[172]; buffer[0][4] =  data[173]; buffer[0][5] =  data[174]; buffer[0][6] =  data[192]; buffer[0][7] =  data[193]; buffer[0][8] =  data[194];

        }
        if (partition == 174) {
            buffer[0][0] =  data[153]; buffer[0][1] =  data[154]; buffer[0][2] =  data[155]; buffer[0][3] =  data[173]; buffer[0][4] =  data[174]; buffer[0][5] =  data[175]; buffer[0][6] =  data[193]; buffer[0][7] =  data[194]; buffer[0][8] =  data[195];

        }
        if (partition == 175) {
            buffer[0][0] =  data[154]; buffer[0][1] =  data[155]; buffer[0][2] =  data[156]; buffer[0][3] =  data[174]; buffer[0][4] =  data[175]; buffer[0][5] =  data[176]; buffer[0][6] =  data[194]; buffer[0][7] =  data[195]; buffer[0][8] =  data[196];

        }
        if (partition == 176) {
            buffer[0][0] =  data[155]; buffer[0][1] =  data[156]; buffer[0][2] =  data[157]; buffer[0][3] =  data[175]; buffer[0][4] =  data[176]; buffer[0][5] =  data[177]; buffer[0][6] =  data[195]; buffer[0][7] =  data[196]; buffer[0][8] =  data[197];

        }
        if (partition == 177) {
            buffer[0][0] =  data[156]; buffer[0][1] =  data[157]; buffer[0][2] =  data[158]; buffer[0][3] =  data[176]; buffer[0][4] =  data[177]; buffer[0][5] =  data[178]; buffer[0][6] =  data[196]; buffer[0][7] =  data[197]; buffer[0][8] =  data[198];

        }
        if (partition == 178) {
            buffer[0][0] =  data[157]; buffer[0][1] =  data[158]; buffer[0][2] =  data[159]; buffer[0][3] =  data[177]; buffer[0][4] =  data[178]; buffer[0][5] =  data[179]; buffer[0][6] =  data[197]; buffer[0][7] =  data[198]; buffer[0][8] =  data[199];

        }
        if (partition == 179) {
            buffer[0][0] =  data[158]; buffer[0][1] =  data[159]; buffer[0][2] =          0; buffer[0][3] =  data[178]; buffer[0][4] =  data[179]; buffer[0][5] =          0; buffer[0][6] =  data[198]; buffer[0][7] =  data[199]; buffer[0][8] =          0;

        }
        if (partition == 180) {
            buffer[0][0] =          0; buffer[0][1] =  data[160]; buffer[0][2] =  data[161]; buffer[0][3] =          0; buffer[0][4] =  data[180]; buffer[0][5] =  data[181]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 181) {
            buffer[0][0] =  data[160]; buffer[0][1] =  data[161]; buffer[0][2] =  data[162]; buffer[0][3] =  data[180]; buffer[0][4] =  data[181]; buffer[0][5] =  data[182]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 182) {
            buffer[0][0] =  data[161]; buffer[0][1] =  data[162]; buffer[0][2] =  data[163]; buffer[0][3] =  data[181]; buffer[0][4] =  data[182]; buffer[0][5] =  data[183]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 183) {
            buffer[0][0] =  data[162]; buffer[0][1] =  data[163]; buffer[0][2] =  data[164]; buffer[0][3] =  data[182]; buffer[0][4] =  data[183]; buffer[0][5] =  data[184]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 184) {
            buffer[0][0] =  data[163]; buffer[0][1] =  data[164]; buffer[0][2] =  data[165]; buffer[0][3] =  data[183]; buffer[0][4] =  data[184]; buffer[0][5] =  data[185]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 185) {
            buffer[0][0] =  data[164]; buffer[0][1] =  data[165]; buffer[0][2] =  data[166]; buffer[0][3] =  data[184]; buffer[0][4] =  data[185]; buffer[0][5] =  data[186]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 186) {
            buffer[0][0] =  data[165]; buffer[0][1] =  data[166]; buffer[0][2] =  data[167]; buffer[0][3] =  data[185]; buffer[0][4] =  data[186]; buffer[0][5] =  data[187]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 187) {
            buffer[0][0] =  data[166]; buffer[0][1] =  data[167]; buffer[0][2] =  data[168]; buffer[0][3] =  data[186]; buffer[0][4] =  data[187]; buffer[0][5] =  data[188]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 188) {
            buffer[0][0] =  data[167]; buffer[0][1] =  data[168]; buffer[0][2] =  data[169]; buffer[0][3] =  data[187]; buffer[0][4] =  data[188]; buffer[0][5] =  data[189]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 189) {
            buffer[0][0] =  data[168]; buffer[0][1] =  data[169]; buffer[0][2] =  data[170]; buffer[0][3] =  data[188]; buffer[0][4] =  data[189]; buffer[0][5] =  data[190]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 190) {
            buffer[0][0] =  data[169]; buffer[0][1] =  data[170]; buffer[0][2] =  data[171]; buffer[0][3] =  data[189]; buffer[0][4] =  data[190]; buffer[0][5] =  data[191]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 191) {
            buffer[0][0] =  data[170]; buffer[0][1] =  data[171]; buffer[0][2] =  data[172]; buffer[0][3] =  data[190]; buffer[0][4] =  data[191]; buffer[0][5] =  data[192]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 192) {
            buffer[0][0] =  data[171]; buffer[0][1] =  data[172]; buffer[0][2] =  data[173]; buffer[0][3] =  data[191]; buffer[0][4] =  data[192]; buffer[0][5] =  data[193]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 193) {
            buffer[0][0] =  data[172]; buffer[0][1] =  data[173]; buffer[0][2] =  data[174]; buffer[0][3] =  data[192]; buffer[0][4] =  data[193]; buffer[0][5] =  data[194]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 194) {
            buffer[0][0] =  data[173]; buffer[0][1] =  data[174]; buffer[0][2] =  data[175]; buffer[0][3] =  data[193]; buffer[0][4] =  data[194]; buffer[0][5] =  data[195]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 195) {
            buffer[0][0] =  data[174]; buffer[0][1] =  data[175]; buffer[0][2] =  data[176]; buffer[0][3] =  data[194]; buffer[0][4] =  data[195]; buffer[0][5] =  data[196]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 196) {
            buffer[0][0] =  data[175]; buffer[0][1] =  data[176]; buffer[0][2] =  data[177]; buffer[0][3] =  data[195]; buffer[0][4] =  data[196]; buffer[0][5] =  data[197]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 197) {
            buffer[0][0] =  data[176]; buffer[0][1] =  data[177]; buffer[0][2] =  data[178]; buffer[0][3] =  data[196]; buffer[0][4] =  data[197]; buffer[0][5] =  data[198]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 198) {
            buffer[0][0] =  data[177]; buffer[0][1] =  data[178]; buffer[0][2] =  data[179]; buffer[0][3] =  data[197]; buffer[0][4] =  data[198]; buffer[0][5] =  data[199]; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
        if (partition == 199) {
            buffer[0][0] =  data[178]; buffer[0][1] =  data[179]; buffer[0][2] =          0; buffer[0][3] =  data[198]; buffer[0][4] =  data[199]; buffer[0][5] =          0; buffer[0][6] =          0; buffer[0][7] =          0; buffer[0][8] =          0;

        }
    }
};

} // namespace nnet

#endif
