#include <iostream>
#include "custom_nn.hpp"

int main() {
    custom_nn::tensor::Tensor<float> t1(10);
    t1[0] = -1.0;
    printf("t1[0]: %f\n", t1[0]);
}
