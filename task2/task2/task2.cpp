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
#include "tensor.hpp"
#include "vgg8.hpp"
#include "data_loader.hpp"

using namespace tensor;
using namespace layer;
using namespace model;


int main() {
    const int shape[3] = { 1, 2, 3 };
    Tensor<float> t1(3, shape);
    //t1[0] = 1.0;

    Model<float>* VGG8 = build_vgg8<float>();
    VGG8->load("vgg8_params.bin");
    VGG8->forward(&t1);
    //data_loader::DataLoader<float> dl;
    //dl.load_data("vgg8_params.bin");
}
