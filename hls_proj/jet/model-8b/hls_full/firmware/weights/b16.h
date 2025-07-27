//Numpy array shape [5]
//Min -0.921875000000
//Max 0.937500000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.921875, 0.937500, -0.218750, -0.359375, 0.750000};

#endif

#endif
