//Numpy array shape [5]
//Min -0.343750000000
//Max 0.250000000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.09375, 0.25000, 0.12500, -0.34375, 0.06250};

#endif

#endif
