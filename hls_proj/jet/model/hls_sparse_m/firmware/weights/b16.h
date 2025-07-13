//Numpy array shape [5]
//Min -0.625000000000
//Max 0.937500000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.62500, 0.21875, 0.12500, -0.34375, 0.93750};

#endif

#endif
