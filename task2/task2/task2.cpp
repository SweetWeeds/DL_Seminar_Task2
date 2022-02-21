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
    int idx = 0;
    DataLoader<float> input_loader;
    input_loader.load_data("input.bin");
    int input_size = 1 * 1 * 28 * 28;
    const int input_shape[4] = { 1, 1, 28, 28 };   // (1, 1, 28, 28)
    float* input_data = new float[1 * 1 * 28 * 28];
    input_loader.copyDataOfRange(input_data, idx, idx + input_size);
    Tensor<float> input_tensor(input_data, 4, input_size, input_shape);
    idx += input_size;

    int label_size = 1 * 10;
    const int label_shape[2] = { 1, 10 };
    float* label_data = new float[1 * 10];
    input_loader.copyDataOfRange(label_data, idx, idx + label_size);
    Tensor<float> label_tensor(label_data, 2, label_size, label_shape);
    
    //t1[0] = 1.0;

    printf("[INFO] Building VGG8 Model... ");
    Model<float>* model = vgg8::build<float>();
    printf("Complete!\n");

    printf("[INFO] Loading Parameters... ");
    model->load("vgg8_params.bin");
    printf("Complete!\n");

    printf("[INFO] Compiling Model... ");
    model->compile(&input_tensor);
    printf("Complete!\n");

    printf("[INFO] Comparing with golden model in Python Numpy...\n");
    model->diff(&input_tensor, &label_tensor, "intermediate_fmap.bin");

    delete model;
    return 0;
}

void printIntro() {
    printf("******************************************\n");
    printf("*                                        *\n");
    printf("*       Deep Learning Seminar Task2      *\n");
    printf("*         VGG8 C++ Implementation        *\n");
    printf("*              Hankyul Kwon              *\n");
    printf("*                                        *\n");
    printf("******************************************\n");
}

int main() {
    printIntro();
    input_test();
    return 0;
}