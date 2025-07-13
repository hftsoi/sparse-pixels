//Numpy array shape [5]
//Min -0.656250000000
//Max 1.375000000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.65625, 0.09375, 0.12500, -0.65625, 1.37500};

#endif

#endif
