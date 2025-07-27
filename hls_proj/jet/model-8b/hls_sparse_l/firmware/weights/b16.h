//Numpy array shape [5]
//Min -0.859375000000
//Max 1.093750000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[5];
#else
bias16_t b16[5] = {-0.328125, -0.859375, 1.093750, -0.375000, 0.578125};

#endif

#endif
