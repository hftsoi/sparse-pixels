//Numpy array shape [3]
//Min -0.125000000000
//Max 0.375000000000
//Number of zeros 0

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
bias2_t b2[3];
#else
bias2_t b2[3] = {-0.125, -0.125, 0.375};

#endif

#endif
