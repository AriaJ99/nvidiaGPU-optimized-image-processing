#include "parser.h"


using namespace std;
void parse(int argc,char* argv[]){
    cxxopts::Options options("GPU-image-processing", "Using Nvidia GPU to perform some image processing functions");
    options.add_options()
    ("i,info", "Extra info that might be needed for some modules", cxxopts::value<int>()->default_value("3"))
    ("f,file", "File path to the image", cxxopts::value<std::string>())
    ("m,module", "Process module that will be applied on the image \n \
      Available modules are (words have be given as option with exact same spelling and cases as mentioned below):\n \
        RGB_to_grayscale (no info needed )\n \
        blur (no info needed / info specifies the kernel size [up to 9, it has to be odd number])\n \
        threshold (info is needed: a positive number smaller than 255 as the threshold to be applied)\n \
        adjust_bright (info is needed: an integer between -255 and 255 to be added to the pixel values)\n \
        edge_detection (no info needed / info specifies the kernel size [only 3*3 kernels are supported for now])\n \
      ", cxxopts::value<std::string>())
    ("c,channel","Image channel",cxxopts::value<std::string>()->default_value("RGB"))
    ("h,help", "Application guide")
  ;
  if (argc==1) {
    std::cout<<options.help()<<std::endl;
    exit(0);
  }
  auto result = options.parse(argc, argv);
  //help handling
  if (result.count("help")) {
    std::cout<<options.help()<<std::endl;
    exit(0);
  }
  run_process(result);
}


void run_process(cxxopts::ParseResult &result){
    string image_file_path=result["f"].as<string>();
    int kernel_size=result["i"].as<int>();
    string module=result["m"].as<string>();
    string channel=result["c"].as<string>();
    //load the image 
    cv::Mat input_image=read_image(image_file_path,channel);
    //run the module
    cv::Mat output_image;
    output_image=run_module(input_image,kernel_size,module);
    //save the output
    std::string output_image_path=processed_image_path_maker(image_file_path,module);
    write_image(output_image,output_image_path);
}
cv::Mat read_image(const string &image_path,const string &channel){
  cv::Mat image;
  if(channel=="RGB")
    image=cv::imread(image_path,cv::IMREAD_COLOR);
  else
    image=cv::imread(image_path,cv::IMREAD_GRAYSCALE);
      if(image.empty())
    {
        std::cerr << "Error: Could not read the image: " << image_path << std::endl;
        exit(-1);
    }
    std::cout<<"Image loaded successfully!!!\n";
    std::cout<<"Image dimensions: "<<image.rows<<" x "<<image.cols<<std::endl;
  
    return image;
}
void write_image(cv::Mat &result_image,std::string &image_path){
    if (result_image.empty()) {
        std::cerr << "Error: Attempted to write an empty image." << std::endl;
        exit(-1);
    }
    bool status = cv::imwrite(image_path, result_image);
    if (!status) {
        std::cerr << "Error: Failed to write the image to " << image_path << std::endl;
    } else {
        std::cout << "Image written successfully to " << image_path << std::endl;
    }
}
std::string processed_image_path_maker(std::string image_file_path,std::string module){
  std::string format;
  for(int i=image_file_path.size()-1;image_file_path[i]!='.';i--){
      format.push_back(image_file_path[i]);
      image_file_path.pop_back();
  }
  //add the "."
  format.push_back('.');
  image_file_path.pop_back();
  //make the path
  std::reverse(format.begin(), format.end());
  image_file_path+="_"+module+"_processed"+format;
  return image_file_path;

}