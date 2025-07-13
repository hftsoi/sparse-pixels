//Numpy array shape [5]
//Min -0.593750000000
//Max 0.843750000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.59375, 0.18750, 0.18750, -0.37500, 0.84375};

#endif

#endif
