/**
 *
 * Deep Learning Seminar Task 2: Custom Neural Netweork Build with C++
 *
 * Written by Kwon, Han Kyul
 * File name: task2.cpp
 * Program date: 2022.02.17.
 *
 * Written on Feb 17, 2022
 * Modified on Feb 17, 2022
 * Modification History:
 *      1. Written by Hankyul Kwon
 *      2. Modified by Han Kyul Kwon on April 27, 2017
 *          (a) Add codes for normal execution. (on parctice hours)
 *
 * Compiler used: MSVC++ 14.16 (Visual Studio 2017 version 15.9)
 *
 */

#include <iostream>
#include <cstdio>
#include "model.hpp"
#include "tensor.hpp"
#include "vgg8.hpp"
#include "data_loader.hpp"

using namespace tensor;
using namespace layer;
using namespace model;

int input_test() {
    DataLoader<float> input_loader;
    input_loader.load_data("input.bin");
    int input_size = 1 * 1 * 28 * 28;
    const int input_shape[4] = { 1, 1, 28, 28 };   // (1, 1, 28, 28)
    float* input_data = new float[1 * 1 * 28 * 28];
    input_loader.copyDataOfRange(input_data, 0, input_size);
    Tensor<float> input_tensor(input_data, 4, input_size, input_shape);
    //t1[0] = 1.0;

    Model<float>* VGG8 = build_vgg8<float>();
    VGG8->load("vgg8_params.bin");
    VGG8->compile(&input_tensor);
    //Tensor<float>* y = VGG8->forward(&input_tensor);
    VGG8->diff(&input_tensor, "intermediate_fmap.bin");
    return 0;
}

int main() {
    input_test();
    return 0;
}