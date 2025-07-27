//Numpy array shape [3]
//Min 0.001953125000
//Max 0.066406250000
//Number of zeros 0

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
bias2_t b2[3];
#else
bias2_t b2[3] = {0.003906250, 0.001953125, 0.066406250};

#endif

#endif
