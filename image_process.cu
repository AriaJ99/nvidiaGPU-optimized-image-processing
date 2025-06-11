#include "image_process.cuh"

cv::Mat run_module(cv::Mat input_image,const std::string &extra_info,const std::string &module){
    cv::Mat output_image;
    cv::Mat temp_image;
    //image dims
    //std::cout<<(int)input_image.data[5]<<std::endl;
    int width=input_image.cols;
    int height=input_image.rows;
    int RGB_size=width*height*3*sizeof(unsigned char);
    int grayscale_size=width*height*sizeof(unsigned char);
    bool is_RGB=input_image.channels();
    //GPU thread dims
    dim3 block_size(32,32);
    dim3 grid_size((width+block_size.x-1)/block_size.x,(height+block_size.y-1)/block_size.y);
    //device memory prepration
    unsigned char *device_input,*device_output;
    //matching the module
    if(module=="RGB_to_grayscale"){
        std::cout<<"converting to grayscale"<<std::endl;
        mem_init(device_input,RGB_size,device_output,grayscale_size,input_image);

        RGB_to_grayscale_kernel<<<grid_size,block_size>>>(device_input,device_output,width,height);
        kernel_error_checker(cudaDeviceSynchronize(),"DeviceSynchronize");

        output_image=mem_to_image(device_output,height,width,grayscale_size,false);

        std::cout<<"convertion to grayscale done!"<<std::endl;
    }
    else if(module=="blur"){
        std::cout<<"start blurring"<<std::endl;

        int kernel_size=std::stoi(extra_info);

        if(is_RGB)
            mem_init(device_input,RGB_size,device_output,RGB_size,input_image);
        else
            mem_init(device_input,grayscale_size,device_output,grayscale_size,input_image);

        blur_kernel<<<grid_size,block_size>>>(device_input,device_output,width,height,kernel_size,is_RGB);
        kernel_error_checker(cudaDeviceSynchronize(),"DeviceSynchronize");
        
        output_image=mem_to_image(device_output,height,width,grayscale_size,is_RGB);

        std::cout<<"image has been blurred!"<<std::endl;
    }
    else if(module=="edge_detection"){
        
        int method;

        if(extra_info=="sobel")
            method=SOBEL;
        else if(extra_info=="prewitt")
            method=PREWITT;
        else if(extra_info=="robert")
            method=ROBERT;
        else
            method=SOBEL;


        std::cout<<"start edge detection"<<std::endl;
        temp_image=input_image;

        unsigned char *vertical_device,*horizontal_device;
        //convert to grayscale if it's RGB
        if(is_RGB){
            std::cout<<"converting to grayscale"<<std::endl;

            mem_init(device_input,RGB_size,device_output,grayscale_size,input_image);

            RGB_to_grayscale_kernel<<<grid_size,block_size>>>(device_input,device_output,width,height);
            kernel_error_checker(cudaDeviceSynchronize(),"DeviceSynchronize");
            
            temp_image=mem_to_image(device_output,height,width,grayscale_size,false);

            kernel_error_checker(cudaFree(device_input),"Free");
            kernel_error_checker(cudaFree(device_output),"Free");
            std::cout<<"convertion to grayscale done!"<<std::endl;
        }

        mem_init(device_input,grayscale_size,device_output,grayscale_size,temp_image);

        kernel_error_checker(cudaMalloc((void**)&vertical_device,grayscale_size),"Malloc");
        kernel_error_checker(cudaMalloc((void**)&horizontal_device,grayscale_size),"Malloc");
        

        edge_detection_kernel<<<grid_size,block_size>>>(device_input,vertical_device,width,height,1,method);
        kernel_error_checker(cudaDeviceSynchronize(),"DeviceSynchronize");
        
        edge_detection_kernel<<<grid_size,block_size>>>(device_input,horizontal_device,width,height,2,method);
        kernel_error_checker(cudaDeviceSynchronize(),"DeviceSynchronize");
        
        matrix_magnitude_kernel<<<grid_size,block_size>>>(device_output,vertical_device,horizontal_device,width,height);
        kernel_error_checker(cudaDeviceSynchronize(),"DeviceSynchronize");

        output_image=mem_to_image(device_output,height,width,grayscale_size,false);
        std::cout<<"edge detection has been done!"<<std::endl;
    }
    else if(module=="threshold"){
        int threshold=std::stoi(extra_info);
        if(!(0<=threshold && threshold<=255))
            threshold=128;

        std::cout<<"Threshold value: "<<threshold<<std::endl;

        temp_image=input_image;
        //convert to grayscale if it's RGB
        if(is_RGB){
            std::cout<<"converting to grayscale"<<std::endl;

            mem_init(device_input,RGB_size,device_output,grayscale_size,input_image);

            RGB_to_grayscale_kernel<<<grid_size,block_size>>>(device_input,device_output,width,height);
            kernel_error_checker(cudaDeviceSynchronize(),"DeviceSynchronize");
            
            temp_image=mem_to_image(device_output,height,width,grayscale_size,false);

            kernel_error_checker(cudaFree(device_input),"Free");
            kernel_error_checker(cudaFree(device_output),"Free");
            std::cout<<"convertion to grayscale done!"<<std::endl;
        }
        std::cout<<"applying the threshold"<<std::endl;
        mem_init(device_input,grayscale_size,device_output,grayscale_size,temp_image);

        thresholding_kernel<<<grid_size,block_size>>>(device_input,device_output,width,height,threshold);
        kernel_error_checker(cudaDeviceSynchronize(),"DeviceSynchronize");

        output_image=mem_to_image(device_output,height,width,grayscale_size,false);
        std::cout<<"Threshold has been applied!"<<std::endl;
    }
    else if(module=="adjust_bright"){
        std::cout<<"start adjusting brightness"<<std::endl;

        int offset=std::stoi(extra_info);

        if(is_RGB)
            mem_init(device_input,RGB_size,device_output,RGB_size,input_image);
        else
            mem_init(device_input,grayscale_size,device_output,grayscale_size,input_image);

        brightness_adjustment_kernel<<<grid_size,block_size>>>(device_input,device_output,width,height,offset,is_RGB);
        kernel_error_checker(cudaDeviceSynchronize(),"DeviceSynchronize");
        
        output_image=mem_to_image(device_output,height,width,grayscale_size,is_RGB);

        std::cout<<"brightness has been adjusted!"<<std::endl;
    }
    else if(module=="edge_sharpening"){
        
        std::cout<<"start edge sharpening"<<std::endl;
 
        if(is_RGB)
            mem_init(device_input,RGB_size,device_output,RGB_size,input_image);
        else
            mem_init(device_input,grayscale_size,device_output,grayscale_size,input_image);

        edge_sharpening_kernel<<<grid_size,block_size>>>(device_input,device_output,width,height,is_RGB);
        kernel_error_checker(cudaDeviceSynchronize(),"DeviceSynchronize");
        
        output_image=mem_to_image(device_output,height,width,grayscale_size,is_RGB);

        std::cout<<"edge sharpening has been done!"<<std::endl;
    }
    else{
        std::cerr<<"Module not found!"<<std::endl;
        exit(-2);
    }
    cudaFree(device_input);
    cudaFree(device_output);
    return output_image;
}
void kernel_error_checker(cudaError_t err, std::string method){
    if (err!=cudaSuccess){
        std::cerr<<"CUDA "<<method<<" error: "<<cudaGetErrorString(err)<<std::endl;
        exit(-1);
    }
}
void mem_init(unsigned char *&input,int input_size,unsigned char *&output,int output_size,cv::Mat &image){
    kernel_error_checker(cudaMalloc((void**)&input,input_size),"Malloc");
    kernel_error_checker(cudaMalloc((void**)&output,output_size),"Malloc");
    kernel_error_checker(cudaMemcpy(input,image.data,input_size,cudaMemcpyHostToDevice),"Memcpy");
}
cv::Mat mem_to_image(unsigned char *input,int height,int width,int size, bool is_RGB){
    cv::Mat temp_img;

    if(is_RGB){
        temp_img=cv::Mat(height,width,CV_8UC3);
        kernel_error_checker(cudaMemcpy(temp_img.data,input,size*3,cudaMemcpyDeviceToHost),"Memcpy");
    }
    else{
        temp_img=cv::Mat(height,width,CV_8UC1);
        kernel_error_checker(cudaMemcpy(temp_img.data,input,size,cudaMemcpyDeviceToHost),"Memcpy");
    }
    
    return temp_img;
}
__device__ int2 find_index() {
    return make_int2(
        blockIdx.x*blockDim.x+threadIdx.x,
        blockIdx.y*blockDim.y+threadIdx.y
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
        //calculate combination of RGB based on human perception of different channels
        output[gray_img_idx]=static_cast<unsigned char>(0.21f*r+0.72f*g+0.07f*b);
    }

}
__global__ void blur_kernel(unsigned char *input,unsigned char *output,int width,int height,int kernel_size,bool is_RGB){
    //TODO handle error for even sized kernels 
    int2 idx=find_index();
    __shared__ float kernel[625];
    //kernel construction
    for(int i=0;i<kernel_size*kernel_size;i++)
        kernel[i]=1.0;
    
    __syncthreads();

    if(idx.x<width && idx.y<height){
        image_kernel_convolution(input,output,width,height,idx.x,idx.y,kernel_size,kernel_size*kernel_size,kernel,is_RGB);
    }
    
}
__global__ void thresholding_kernel(unsigned char *input,unsigned char *output,int width,int height,int threshold){
    int2 idx=find_index();
    if(idx.x<width && idx.y<height){
        int grayscale_img_idx=(idx.y*width+idx.x);
        if(static_cast<unsigned char>(threshold)<=input[grayscale_img_idx])
            output[grayscale_img_idx]=static_cast<unsigned char>(255);
        else
            output[grayscale_img_idx]=static_cast<unsigned char>(0);
    }
}
__global__ void brightness_adjustment_kernel(unsigned char *input,unsigned char *output,int width,int height,int offset,bool is_RGB){
    int2 idx=find_index();
    if(is_RGB){
        int RGB_img_idx=(idx.y*width+idx.x)*3;
        if(idx.x<width && idx.y<height){
                int grayscale_img_idx=(idx.y*width+idx.x);
                if(offset>=0){
                    output[RGB_img_idx]=min(input[RGB_img_idx]+offset,255);
                    output[RGB_img_idx+1]=min(input[RGB_img_idx+1]+offset,255);
                    output[RGB_img_idx+2]=min(input[RGB_img_idx+2]+offset,255);
                }
                else{
                    output[RGB_img_idx]=max(input[RGB_img_idx]-offset,0);
                    output[RGB_img_idx+1]=max(input[RGB_img_idx+1]-offset,0);
                    output[RGB_img_idx+2]=max(input[RGB_img_idx+2]-offset,0);
                }
            }
    }
    else{
        int grayscale_img_idx=idx.y*width+idx.x;

        if(idx.x<width && idx.y<height){
                int grayscale_img_idx=(idx.y*width+idx.x);
                if(offset>=0)
                    output[grayscale_img_idx]=min(input[grayscale_img_idx]+offset,255);
                else
                    output[grayscale_img_idx]=max(input[grayscale_img_idx]-offset,0);

            }
    }
}
__device__ void image_kernel_convolution(unsigned char *input,unsigned char *output,int width,int height,int x,int y,int kernel_size,int kernel_weight,float *kernel,bool is_RGB){
    //TODO pixel overflow handling

    int RGB_idx=(y*width+x)*3;
    int grayscale_idx=y*width+x;

    float sum_r=0.0,sum_g=0.0,sum_b=0.0,sum_gray=0.0;
    for(int dy=0;dy<kernel_size;dy++)
        for(int dx=0;dx<kernel_size;dx++){
            int nx=x+dx-kernel_size/2;
            int ny=y+dy-kernel_size/2;

            if(nx>=0 && nx<width && ny>=0 && ny<height){
                float weight=kernel[dy*kernel_size+dx];
                if(is_RGB){
                    int nidx=(ny*width+nx)*3;

                    sum_r+=input[nidx]*weight;
                    sum_g+=input[nidx+1]*weight;
                    sum_b+=input[nidx+2]*weight;
                }
                else{
                    int nidx=ny*width+nx;

                    sum_gray+=input[nidx]*weight;
                }
                
            }
        }
    if(is_RGB){
        output[RGB_idx]=static_cast<unsigned char>(max(0,min(static_cast<int>(sum_r/kernel_weight),255)));
        output[RGB_idx+1]=static_cast<unsigned char>(max(0,min(255,static_cast<int>(sum_g/kernel_weight))));
        output[RGB_idx+2]=static_cast<unsigned char>(max(0,min(static_cast<int>(sum_b/kernel_weight),255)));
    }
    else{
        output[grayscale_idx]=static_cast<unsigned char>(max(0,min(static_cast<int>(sum_gray/kernel_weight),255)));
    }    

}
__global__ void edge_detection_kernel(unsigned char *input,unsigned char *output,int width,int height,int direction,int method){
    //TODO handle error for even sized kernels 
    int2 idx=find_index();

    if(idx.x>=width || idx.y>=height)
        return;
    __shared__ float kernel_y[9],kernel_x[9];
    //kernel construction
    if(method==SOBEL){
        //kernel values
        // Sobel x -1   0   1
        //         -2   0   2
        //         -1   0   1
        kernel_x[0]=-1,   kernel_x[1]=0,    kernel_x[2]=1;  
        kernel_x[3]=-2,   kernel_x[4]=0,    kernel_x[5]=2;
        kernel_x[6]=-1,   kernel_x[7]=0,    kernel_x[8]=1;

        // Sobel y -1  -2  -1
        //          0   0   0
        //          1   2   1
        kernel_y[0]=-1,   kernel_y[1]=-2,    kernel_y[2]=-1;  
        kernel_y[3]=0,    kernel_y[4]=0,     kernel_y[5]=0;
        kernel_y[6]=1,    kernel_y[7]=2,     kernel_y[8]=1;

    }
    else if(method==PREWITT){
        //kernel values
        // Prewitt x 1   0  -1
        //           1   0  -1
        //           1   0  -1
        kernel_x[0]=1,   kernel_x[1]=0,    kernel_x[2]=-1;  
        kernel_x[3]=1,   kernel_x[4]=0,    kernel_x[5]=-1;
        kernel_x[6]=1,   kernel_x[7]=0,    kernel_x[8]=-1;

        // Prewitt y  1   1   1
        //            0   0   0
        //           -1  -1  -1
        kernel_y[0]=1,   kernel_y[1]=1,    kernel_y[2]=1;  
        kernel_y[3]=0,   kernel_y[4]=0,    kernel_y[5]=0;
        kernel_y[6]=-1,  kernel_y[7]=-1,   kernel_y[8]=-1;

    }
    else if(method==ROBERT){
        //kernel values
        // Robert x  0   0   0
        //           0   1   0
        //           0   0   -1
        kernel_x[0]=0,   kernel_x[1]=0,    kernel_x[2]=0;  
        kernel_x[3]=0,   kernel_x[4]=1,    kernel_x[5]=0;
        kernel_x[6]=0,   kernel_x[7]=0,    kernel_x[8]=-1;
       // Robert x  0   0   0
        //          0   0   1
        //          0  -1   0
        kernel_y[0]=0,   kernel_y[1]=0,    kernel_y[2]=0;  
        kernel_y[3]=0,   kernel_y[4]=0,    kernel_y[5]=1;
        kernel_y[6]=0,   kernel_y[7]=-1,   kernel_y[8]=0;

    }
    __syncthreads();


    if(direction==1)
        image_kernel_convolution(input,output,width,height,idx.x,idx.y,3,1,kernel_x,false);
    else if(direction==2)
        image_kernel_convolution(input,output,width,height,idx.x,idx.y,3,1,kernel_y,false);




}
__global__  void matrix_magnitude_kernel(unsigned char *destination,unsigned char *matrix_1,unsigned char *matrix_2,int width,int height){
    int2 idx=find_index();
    if(idx.y<height && idx.x<width){
        int index=idx.y*width+idx.x;
        destination[index]=min(255,static_cast<int>(sqrtf(matrix_1[index]*matrix_1[index]+matrix_2[index]*matrix_2[index])));
    }
}
__global__ void edge_sharpening_kernel(unsigned char *input,unsigned char *output,int width,int height,bool is_RGB){
    int2 idx=find_index();

    if(idx.x>=width || idx.y>=height)
        return;

    __shared__ float kernel[9];

    //kernel values
    // Laplacian    0   -1   0
    //             -1    5  -1
    //              0   -1   0
    kernel[0]=0,   kernel[1]=-1,    kernel[2]=0;  
    kernel[3]=-1,   kernel[4]=5,    kernel[5]=-1;
    kernel[6]=0,   kernel[7]=-1,    kernel[8]=0;
    
    __syncthreads();
    

    image_kernel_convolution(input,output,width,height,idx.x,idx.y,3,1,kernel,is_RGB);

    

}