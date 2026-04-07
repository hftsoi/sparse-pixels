//Numpy array shape [3]
//Min 0.625000000000
//Max 0.750000000000
//Number of zeros 0

#ifndef B8_H_
#define B8_H_

#ifndef __SYNTHESIS__
conv2_bias_t b8[3];
#else
conv2_bias_t b8[3] = {0.6875, 0.7500, 0.6250};

#endif

#endif
