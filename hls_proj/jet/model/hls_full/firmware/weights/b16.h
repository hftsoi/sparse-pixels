//Numpy array shape [5]
//Min -0.218750000000
//Max 0.156250000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.21875, 0.15625, -0.03125, 0.12500, 0.03125};

#endif

#endif
