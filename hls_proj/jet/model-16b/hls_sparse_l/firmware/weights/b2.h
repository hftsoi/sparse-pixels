//Numpy array shape [3]
//Min -0.046875000000
//Max 0.042968750000
//Number of zeros 0

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
bias2_t b2[3];
#else
bias2_t b2[3] = {0.001953125, 0.042968750, -0.046875000};

#endif

#endif
