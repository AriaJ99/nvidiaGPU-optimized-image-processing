#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H
#include <iostream>
#include <utility>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime.h>
cv::Mat run_module(cv::Mat input_image,const int &extra_info,const std::string &module);
__global__ void RGB_to_grayscale_kernel(unsigned char *input,unsigned char *output,int width,int height);
void kernel_error_checker(cudaError_t err, std::string method);
__device__ int2 find_index();
__global__ void blur_kernel(unsigned char *input,unsigned char *output,int width,int height,int kernel_size,bool is_RGB);
__global__ void thresholding_kernel(unsigned char *input,unsigned char *output,int width,int height,int threshold);
__global__ void brightness_adjustment_kernel(unsigned char *input,unsigned char *output,int width,int height,int offset,bool is_RGB);
__device__ void image_kernel_convolution(unsigned char *input,unsigned char *output,int width,int height,int x,int y,int kernel_size,int kernel_weight,float *kernel,bool is_RGB);
__global__ void edge_detection_kernel(unsigned char *input,unsigned char *output,int width,int height,int kernel_size);
void mem_init(unsigned char *&input,int input_size,unsigned char *&output,int output_size,cv::Mat &image);
cv::Mat mem_to_image(unsigned char *input,int height,int width,int size, bool is_RGB);
#endif