//Numpy array shape [10]
//Min -0.500000000000
//Max 0.500000000000
//Number of zeros 3

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
dense2_bias_t b16[10];
#else
dense2_bias_t b16[10] = {-0.25, 0.50, 0.25, 0.00, 0.00, 0.50, -0.25, 0.00, -0.50, -0.25};

#endif

#endif
