#include <iostream>
#include <cstdio>
#include "custom_nn.hpp"

using namespace custom_nn::tensor;

int main() {
    int t1_size[2] = { 1, 2 };
    Tensor<float> t1(2, t1_size);
    printf("t1 size:%d, %f, %f\n", t1.getSize(), t1[0], t1[1]);

}
