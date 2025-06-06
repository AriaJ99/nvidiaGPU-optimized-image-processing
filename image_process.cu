#include "image_process.cuh"

cv::Mat run_module(cv::Mat input_image,const int &kernel_size,const std::string &module){
    cv::Mat output_image;
    //image dims
    int width=input_image.cols;
    int height=input_image.rows;
    int RGB_size=width*height*3*sizeof(unsigned char);
    int grayscale_size=width*height*sizeof(unsigned char);
    //GPU thread dims
    dim3 block_size(32,32);
    dim3 grid_size((width+block_size.x-1)/block_size.x,(height+block_size.y-1)/block_size.y);
    //device memory prepration
    unsigned char *device_input,*device_output;
    //matching the module
    if(module=="RGB_to_grayscale"){
        cudaMalloc((void**)&device_input,RGB_size);
        cudaMalloc((void**)&device_output,grayscale_size);
        cudaMemcpy(device_input,input_image.data,RGB_size,cudaMemcpyHostToDevice);
        RGB_to_grayscale_kernel<<<grid_size,block_size>>>(device_input,device_output,width,height);
        cudaDeviceSynchronize();
        output_image=cv::Mat(height,width,CV_8UC1);
        cudaMemcpy(output_image.data,device_output,grayscale_size,cudaMemcpyDeviceToHost);
    }
    else if(module=="blur"){
        cudaMalloc((void**)&device_input,RGB_size);
        cudaMalloc((void**)&device_output,RGB_size);
        cudaMemcpy(device_input,input_image.data,RGB_size,cudaMemcpyHostToDevice);
        //blur_kernel<<<grid_size,block_size>>>(device_input,device_output,width,height);
        cudaDeviceSynchronize();
        output_image=cv::Mat(height,width,CV_8UC3);
        cudaMemcpy(output_image.data,device_output,RGB_size,cudaMemcpyDeviceToHost);
    }
    cudaFree(device_input);
    cudaFree(device_output);
    return output_image;
}
__device__ int2 find_index() {
    return make_int2(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    );
}
__global__ void RGB_to_grayscale_kernel(unsigned char *input,unsigned char *output,int width,int height){
    int2 idx=find_index();
    if(idx.x<width && idx.y<height){
        int rgb_img_idx=(idx.y*width+idx.x)*3;
        int gray_img_idx=(idx.y*width+idx.x);
        unsigned char r=input[rgb_img_idx];
        unsigned char g=input[rgb_img_idx+1];
        unsigned char b=input[rgb_img_idx+2];

        output[gray_img_idx]=static_cast<unsigned char>(0.21f*r+0.72f*g+0.07f*b);
    }

}