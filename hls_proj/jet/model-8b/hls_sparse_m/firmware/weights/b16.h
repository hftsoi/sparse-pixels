//Numpy array shape [5]
//Min -0.921875000000
//Max 0.968750000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.921875, 0.046875, 0.562500, -0.406250, 0.968750};

#endif

#endif
