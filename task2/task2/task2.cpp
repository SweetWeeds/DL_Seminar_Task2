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

void padding_test();
void indexing_test();
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
    float maxDiff = VGG8->diff(&input_tensor, "intermediate_fmap.bin");
    printf("[INFO] max diff: %f\n", maxDiff);
    //data_loader::DataLoader<float> dl;
    //dl.load_data("vgg8_params.bin");
    return 0;
}

int main() {
    input_test();
    return 0;
}
// Padding
template <typename T>
Tensor<T>* pad(Tensor<T> *p_x, const int p) {
    Tensor<T>& x = *p_x;
    const int* x_shape = x.getShape();
    const int B = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
    const int padded_shape[] = { B, C, H + 2 * p, W + 2 * p };
    Tensor<T> *p_padded_x = new Tensor<T>(4, padded_shape);
    Tensor<T>& padded_x = *p_padded_x;
    padded_x.dataInit();
    int padded_x_index[4] = { 0, };
    int x_index[4] = { 0, };
    for (int b = 0; b < x_shape[0]; b++) {
        padded_x_index[0] = b;
        x_index[0] = b;
        for (int c = 0; c < x_shape[1]; c++) {
            padded_x_index[1] = c;
            x_index[1] = c;
            for (int h = 0; h < x_shape[2]; h++) {
                padded_x_index[2] = h + p;
                x_index[2] = h;
                for (int w = 0; w < x_shape[3]; w++) {
                    padded_x_index[3] = w + p;
                    x_index[3] = w;
                    padded_x[index_calc(4, padded_shape, padded_x_index)] = x[index_calc(4, x_shape, x_index)];
                }
            }
        }
    }
    return p_padded_x;
}


void indexing_test() {
    int shape[2] = { 10,5 };
    int idx[2] = { 0, };
    for (int i = 0; i < 10; i++) {
        idx[0] = i;
        for (int j = 0; j < 5; j++) {
            idx[1] = j;
            int ret = index_calc(2, shape, idx);
            printf("%d\n", ret);
        }
    }
}

void padding_test() {
    const int input_shape[4] = { 1, 1, 28, 28 };
    Tensor<float> input_tensor(4, input_shape);
    for (int i = 1; i <= 28; i++) {
        for (int j = 1; j <= 28; j++) {
            input_tensor[(i-1) * 28 + (j-1)] = (float) i * j;
        }
    }
    Tensor<float>* padded_tensor = pad<float>(&input_tensor, 1);
    return;
}