//Numpy array shape [5]
//Min -0.957031250000
//Max 1.414062500000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.064453125, -0.957031250, 0.285156250, -0.396484375, 1.414062500};

#endif

#endif
