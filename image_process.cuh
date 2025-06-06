#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H
#include <iostream>
#include <utility>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime.h>
cv::Mat run_module(cv::Mat input_image,const int &kernel_size,const std::string &module);
__global__ void RGB_to_grayscale_kernel(unsigned char *input,unsigned char *output,int width,int height);
__device__ int2 find_index();
#endif