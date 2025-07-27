//Numpy array shape [5]
//Min -1.125000000000
//Max 1.468750000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-1.125000, -0.203125, -0.203125, 0.296875, 1.468750};

#endif

#endif
