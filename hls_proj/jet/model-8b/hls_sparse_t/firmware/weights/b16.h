//Numpy array shape [5]
//Min -0.890625000000
//Max 1.171875000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.890625, -0.437500, -0.078125, 0.343750, 1.171875};

#endif

#endif
