//Numpy array shape [3]
//Min 0.000000000000
//Max 0.500000000000
//Number of zeros 1

#ifndef B7_H_
#define B7_H_

#ifndef __SYNTHESIS__
conv2_bias_t b7[3];
#else
conv2_bias_t b7[3] = {0.25, 0.50, 0.00};

#endif

#endif
