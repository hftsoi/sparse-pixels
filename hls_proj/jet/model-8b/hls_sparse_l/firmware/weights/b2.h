//Numpy array shape [3]
//Min -0.031250000000
//Max 0.109375000000
//Number of zeros 0

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
bias2_t b2[3];
#else
bias2_t b2[3] = {0.015625, -0.031250, 0.109375};

#endif

#endif
