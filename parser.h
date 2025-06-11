#ifndef PARSER_H
#define PARSER_H
#include "image_process.cuh"
#include <iostream>
#include <utility>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "cxxopts.hpp"
void parse(int argc,char* argv[]);
void run_process(cxxopts::ParseResult &result);
cv::Mat read_image(const std::string &image_path,const int &channel);
void write_image(cv::Mat &result_image,std::string &image_path);
std::string processed_image_path_maker(std::string image_file_path,std::string module);
#endif