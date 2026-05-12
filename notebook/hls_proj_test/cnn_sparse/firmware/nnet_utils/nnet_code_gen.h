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

template<typename input_t, typename output_t>
void conv1_iq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[3]);
    out[4] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[4]);
    out[5] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[5]);
    out[6] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[6]);
    out[7] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[7]);
    out[8] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[8]);
    out[9] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[9]);
    out[10] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[10]);
    out[11] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[11]);
    out[12] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[12]);
    out[13] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[13]);
    out[14] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[14]);
    out[15] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[15]);
    out[16] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[16]);
    out[17] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[17]);
    out[18] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[18]);
    out[19] = ap_fixed<10,3,AP_RND,AP_WRAP>(inp[19]);
}

template<typename input_t, typename output_t>
void conv2_iq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[3]);
    out[4] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[4]);
    out[5] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[5]);
    out[6] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[6]);
    out[7] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[7]);
    out[8] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[8]);
    out[9] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[9]);
    out[10] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[10]);
    out[11] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[11]);
    out[12] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[12]);
    out[13] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[13]);
    out[14] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[14]);
    out[15] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[15]);
    out[16] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[16]);
    out[17] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[17]);
    out[18] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[18]);
    out[19] = ap_fixed<9,2,AP_RND,AP_WRAP>(inp[19]);
}

template<typename input_t, typename output_t>
void dense1_iq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[3]);
    out[4] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[4]);
    out[5] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[5]);
    out[6] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[6]);
    out[7] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[7]);
    out[8] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[8]);
    out[9] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[9]);
    out[10] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[10]);
    out[11] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[11]);
    out[12] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[12]);
    out[13] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[13]);
    out[14] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[14]);
    out[15] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[15]);
    out[16] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[16]);
    out[17] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[17]);
    out[18] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[18]);
    out[19] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[19]);
    out[20] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[20]);
    out[21] = ap_fixed<2,1,AP_RND,AP_WRAP>(inp[21]);
    out[22] = ap_fixed<3,2,AP_RND,AP_WRAP>(inp[22]);
    out[23] = ap_fixed<3,2,AP_RND,AP_WRAP>(inp[23]);
    out[24] = ap_fixed<5,3,AP_RND,AP_WRAP>(inp[24]);
    out[25] = ap_fixed<6,3,AP_RND,AP_WRAP>(inp[25]);
    out[26] = ap_fixed<5,3,AP_RND,AP_WRAP>(inp[26]);
    out[27] = ap_fixed<5,3,AP_RND,AP_WRAP>(inp[27]);
    out[28] = ap_fixed<6,3,AP_RND,AP_WRAP>(inp[28]);
    out[29] = ap_fixed<5,3,AP_RND,AP_WRAP>(inp[29]);
    out[30] = ap_fixed<3,2,AP_RND,AP_WRAP>(inp[30]);
    out[31] = ap_fixed<4,2,AP_RND,AP_WRAP>(inp[31]);
    out[32] = ap_fixed<3,2,AP_RND,AP_WRAP>(inp[32]);
    out[33] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[33]);
    out[34] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[34]);
    out[35] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[35]);
    out[36] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[36]);
    out[37] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[37]);
    out[38] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[38]);
    out[39] = ap_fixed<4,3,AP_RND,AP_WRAP>(inp[39]);
    out[40] = ap_fixed<4,3,AP_RND,AP_WRAP>(inp[40]);
    out[41] = ap_fixed<5,3,AP_RND,AP_WRAP>(inp[41]);
    out[42] = ap_fixed<4,3,AP_RND,AP_WRAP>(inp[42]);
    out[43] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[43]);
    out[44] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[44]);
    out[45] = ap_fixed<5,3,AP_RND,AP_WRAP>(inp[45]);
    out[46] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[46]);
    out[47] = ap_fixed<5,3,AP_RND,AP_WRAP>(inp[47]);
    out[48] = ap_fixed<3,2,AP_RND,AP_WRAP>(inp[48]);
    out[49] = ap_fixed<4,3,AP_RND,AP_WRAP>(inp[49]);
    out[50] = ap_fixed<4,3,AP_RND,AP_WRAP>(inp[50]);
    out[51] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[51]);
    out[52] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[52]);
    out[53] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[53]);
    out[54] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[54]);
    out[55] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[55]);
    out[56] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[56]);
    out[57] = ap_fixed<5,3,AP_RND,AP_WRAP>(inp[57]);
    out[58] = ap_fixed<4,3,AP_RND,AP_WRAP>(inp[58]);
    out[59] = ap_fixed<4,3,AP_RND,AP_WRAP>(inp[59]);
    out[60] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[60]);
    out[61] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[61]);
    out[62] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[62]);
    out[63] = ap_fixed<4,3,AP_RND,AP_WRAP>(inp[63]);
    out[64] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[64]);
    out[65] = ap_fixed<4,3,AP_RND,AP_WRAP>(inp[65]);
    out[66] = ap_fixed<3,2,AP_RND,AP_WRAP>(inp[66]);
    out[67] = ap_fixed<4,3,AP_RND,AP_WRAP>(inp[67]);
    out[68] = ap_fixed<3,2,AP_RND,AP_WRAP>(inp[68]);
    out[69] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[69]);
    out[70] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[70]);
    out[71] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[71]);
    out[72] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[72]);
    out[73] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[73]);
    out[74] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[74]);
    out[75] = ap_fixed<3,2,AP_RND,AP_WRAP>(inp[75]);
    out[76] = ap_fixed<3,2,AP_RND,AP_WRAP>(inp[76]);
    out[77] = ap_fixed<2,2,AP_RND,AP_WRAP>(inp[77]);
    out[78] = ap_fixed<5,3,AP_RND,AP_WRAP>(inp[78]);
    out[79] = ap_fixed<5,3,AP_RND,AP_WRAP>(inp[79]);
    out[80] = ap_fixed<4,2,AP_RND,AP_WRAP>(inp[80]);
    out[81] = ap_fixed<4,2,AP_RND,AP_WRAP>(inp[81]);
    out[82] = ap_fixed<5,3,AP_RND,AP_WRAP>(inp[82]);
    out[83] = ap_fixed<4,2,AP_RND,AP_WRAP>(inp[83]);
    out[84] = ap_fixed<2,0,AP_RND,AP_WRAP>(inp[84]);
    out[85] = ap_fixed<3,2,AP_RND,AP_WRAP>(inp[85]);
    out[86] = ap_fixed<2,1,AP_RND,AP_WRAP>(inp[86]);
    out[87] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[87]);
    out[88] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[88]);
    out[89] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[89]);
    out[90] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[90]);
    out[91] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[91]);
    out[92] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[92]);
    out[93] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[93]);
    out[94] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[94]);
    out[95] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[95]);
    out[96] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[96]);
    out[97] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[97]);
    out[98] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[98]);
    out[99] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[99]);
    out[100] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[100]);
    out[101] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[101]);
    out[102] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[102]);
    out[103] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[103]);
    out[104] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[104]);
    out[105] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[105]);
    out[106] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[106]);
    out[107] = ap_fixed<1,-6,AP_RND,AP_WRAP>(inp[107]);
}

template<typename input_t, typename output_t>
void dense2_iq(input_t *inp, output_t *out) {
    #pragma HLS INLINE

    out[0] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[0]);
    out[1] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[1]);
    out[2] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[2]);
    out[3] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[3]);
    out[4] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[4]);
    out[5] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[5]);
    out[6] = 0;
    out[7] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[7]);
    out[8] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[8]);
    out[9] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[9]);
    out[10] = 0;
    out[11] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[11]);
    out[12] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[12]);
    out[13] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[13]);
    out[14] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[14]);
    out[15] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[15]);
    out[16] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[16]);
    out[17] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[17]);
    out[18] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[18]);
    out[19] = ap_fixed<4,3,AP_RND,AP_WRAP>(inp[19]);
    out[20] = ap_fixed<6,5,AP_RND,AP_WRAP>(inp[20]);
    out[21] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[21]);
    out[22] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[22]);
    out[23] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[23]);
    out[24] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[24]);
    out[25] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[25]);
    out[26] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[26]);
    out[27] = ap_fixed<7,5,AP_RND,AP_WRAP>(inp[27]);
    out[28] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[28]);
    out[29] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[29]);
    out[30] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[30]);
    out[31] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[31]);
    out[32] = ap_fixed<5,4,AP_RND,AP_WRAP>(inp[32]);
    out[33] = ap_fixed<5,3,AP_RND,AP_WRAP>(inp[33]);
    out[34] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[34]);
    out[35] = ap_fixed<6,4,AP_RND,AP_WRAP>(inp[35]);
}

} // namespace nnet

#endif
