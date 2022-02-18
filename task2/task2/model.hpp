/**
 *
 * Deep Learning Seminar Task 2: Custom Neural Netweork Build with C++
 *
 * Written by Kwon, Han Kyul
 * File name: model.cpp
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
#ifndef MODEL_H
#define MODEL_H

#define MAX_LAYER_NUM 100

#include <string.h>
#include <assert.h>
#include "data_loader.hpp"
#include "tensor.hpp"
#include "layer.hpp"

using namespace tensor;
using namespace layer;

namespace model {
    template <typename T>
    class Model {
    private:
        Layer<T>* layers[MAX_LAYER_NUM];
        int num_layers = 0;
    public:
        Model(Layer<T>* layers[], int num_layers) : num_layers(num_layers) {
            memcpy(this->layers, layers, num_layers * sizeof(Layer<T>*));
        }

        ~Model() {
            for (int i = 0; i < num_layers; i++) {
                delete this->layers[i];
            }
        }

        Tensor<T>* forward(Tensor<T>* x) {
            for (int i = 0; i < num_layers; i++) {
                x = this->layers[i]->forward(x);
            }
            return x;
        }

        Tensor<T>* backward(Tensor<T>* dout) {
            for (int i = num_layers - 1; i >= 0; i--) {
                dout = this->layers[i]->backward(dout);
            }
            return dout;
        }

        bool save(const char* fn) {

        }

        bool load(const char* fn) {
            std::ifstream is(fn, std::ifstream::binary);
            T* params = nullptr;
            if (is) {
                is.seekg(0, is.end);
                int length = (int)is.tellg();
                is.seekg(0, is.beg);
                params = new T[length];
                assert(params);
                is.read((char*)params, length);
                is.close();
            }
            if (params == nullptr) delete params;
            return true;
        }
    };


}

#endif
/** End of moidel.hpp **/