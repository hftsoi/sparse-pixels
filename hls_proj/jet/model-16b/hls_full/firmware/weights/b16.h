//Numpy array shape [5]
//Min -0.341796875000
//Max 0.494140625000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.171875000, 0.058593750, 0.076171875, -0.341796875, 0.494140625};

#endif

#endif
