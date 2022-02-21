/**
 *
 * Deep Learning Seminar Task 2: Custom Neural Netweork Build with C++
 *
 * Written by Kwon, Han Kyul
 * File name: vgg8.hpp
 * Program date: 2022.02.17.
 *
 * Written on Feb 17, 2022
 * Modified on Feb 17, 2022
 * Modification History:
 *      1. Written by Hankyul Kwon
 *      2. Modified by Han Kyul Kwon on Feb 17, 2022
 *          (a) Designed top classes.
 *
 * Compiler used: MSVC++ 14.16 (Visual Studio 2017 version 15.9)
 */

#ifndef VGG8_H
#define VGG8_H

#include "common.hpp"
#include "model.hpp"


namespace vgg8 {
    using namespace model;

    template <typename T>
    Model<T>* build() {
        Layer<float>* layers[17] = { nullptr, };

        // Layer 1 (B, 1, 28, 28) -> (B, 32, 28, 28)
        layers[0]  = (Layer<float>*) new Conv2D<T>(1, 32, 3, 1, 1, "L1_C");
        layers[1]  = (Layer<float>*) new ReLU<T>("L1_R");

        // Layer2 (B, 32, 28, 28) -> (B, 64, 14, 14)
        layers[2]  = (Layer<float>*) new Conv2D<T>(32, 64, 3, 1, 1, "L2_C");
        layers[3]  = (Layer<float>*) new ReLU<T>("L2_R");
        layers[4]  = (Layer<float>*) new MaxPool2D<T>(2, 2, "L2_M");

        // Layer 3 (B, 64, 14, 14) -> (B, 64, 14, 14)
        layers[5]  = (Layer<float>*) new Conv2D<T>(64, 64, 3, 1, 1, "L3_C");
        layers[6]  = (Layer<float>*) new ReLU<T>("L3_R");

        // Layer 4 (B, 64, 14, 14) -> (B, 128, 7, 7)
        layers[7]  = (Layer<float>*) new Conv2D<T>(64, 128, 3, 1, 1, "L4_C");
        layers[8]  = (Layer<float>*) new ReLU<T>("L4_R");
        layers[9]  = (Layer<float>*) new MaxPool2D<T>(2, 2, "L4_M");

        // Layer 5 (B, 128, 7, 7) -> (B, 256, 7, 7)
        layers[10] = (Layer<float>*) new Conv2D<T>(128, 256, 3, 1, 1, "L5_C");
        layers[11] = (Layer<float>*) new ReLU<T>("L5_R");

        // Layer 6 (B, 256, 7, 7) -> (B, 256, 7, 7)
        layers[12] = (Layer<float>*) new Conv2D<T>(256, 256, 3, 1, 1, "L6_C");
        layers[13] = (Layer<float>*) new ReLU<T>("L6_R");

        // Layer 7 (B, 256*7*7) -> (B, 256)
        layers[14] = (Layer<float>*) new FullyConnected<T>(256 * 7 * 7, 256, "L7_FC");
        layers[15] = (Layer<float>*) new ReLU<T>("L7_R");

        // Layer 8 (B, 256) -> (B, 10)
        layers[16] = (Layer<float>*) new FullyConnected<T>(256, 10, "L8_FC");

        Criterion<float>* criterion = (Criterion<float>*) new SoftmaxWithLoss<float>("softmax");

        Model<T>* model = new Model<T>(layers, criterion, 17, "VGG8");
        return model;
    }
}
#endif
 /** End of tensor.hpp **/