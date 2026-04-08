//Numpy array shape [3]
//Min 0.375000000000
//Max 0.687500000000
//Number of zeros 0

#ifndef B8_H_
#define B8_H_

#ifndef __SYNTHESIS__
conv2_bias_t b8[3];
#else
conv2_bias_t b8[3] = {0.6875, 0.5000, 0.3750};

#endif

#endif
