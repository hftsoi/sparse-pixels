//Numpy array shape [5]
//Min -0.597656250000
//Max 1.808593750000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.597656250, -0.435546875, -0.406250000, -0.105468750, 1.808593750};

#endif

#endif
