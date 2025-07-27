//Numpy array shape [5]
//Min -0.771484375000
//Max 1.078125000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.199218750, 0.136718750, -0.029296875, -0.771484375, 1.078125000};

#endif

#endif
