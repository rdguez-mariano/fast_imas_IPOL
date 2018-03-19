// Authors: Unknown. Please, if you are the author of this file, or if you
// know who are the authors of this file, let us know, so we can give the
// adequate credits and/or get the adequate authorizations.


#ifndef _SIMU_TILTS_H_
#define _SIMU_TILTS_H_

void simulate_digital_tilt(const std::vector<float>& image, int width, int height, std::vector<float>& image_to_return, int& width_t, int& height_t, float theta, float t,float sigma);
void GaussianBlur1D(std::vector<float>& image, int width, int height, float sigma, int flag_dir);
#endif
