//Numpy array shape [5]
//Min -0.312500000000
//Max 0.406250000000
//Number of zeros 1

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.31250, -0.06250, 0.03125, 0.00000, 0.40625};

#endif

#endif
