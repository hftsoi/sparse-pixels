//Numpy array shape [5]
//Min -0.437500000000
//Max 0.500000000000
//Number of zeros 1

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.25000, 0.25000, 0.00000, -0.43750, 0.50000};

#endif

#endif
