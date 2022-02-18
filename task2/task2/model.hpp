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
using namespace data_loader;

namespace model {
    template <typename T>
    class Model {
    private:
        Layer<T>* layers[MAX_LAYER_NUM];
        int num_layers = 0;
        DataLoader<T> dl;
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

        int save(const char* fn) {
            return 0;
        }

        int load(const char* fn) {
            if (dl.load_data(fn)) {
                #ifdef MODEL_DEBUG
                printf("[DEBUG:Model:load] Failed to load data (%s).\n", fn);
                #endif
                return 1;
            }
            const int* w_shape;
            const int* b_shape;
            int w_size;
            int b_size;
            int w_dim;
            int b_dim;
            int idx = 0;
            for (int i = 0; i < num_layers; i++) {
                w_shape = this->layers[i]->getWeightShape();
                b_shape = this->layers[i]->getBiasShape();
                if (w_shape == nullptr || b_shape == nullptr) {
                    #ifdef MODEL_DEBUG
                    printf("[DEBUG:Model:load] %s has empty params.\n", this->layers[i]->getName());
                    #endif
                    continue;
                }
                w_size  = this->layers[i]->getWeightSize();
                b_size  = this->layers[i]->getBiasSize();
                w_dim   = this->layers[i]->getWeightDim();
                b_dim   = this->layers[i]->getBiasDim();
                Tensor<T> W(w_dim, w_shape), b(b_dim, b_shape);
                if (dl.copyDataOfRange(W.getData(), idx, idx + w_size))
                    printf("[ERROR:Model:load] Failed to load %s layer. (idx:%d, w_size:%d)\n", layers[i]->getName(), idx, w_size);
                idx += w_size;
                if (dl.copyDataOfRange(b.getData(), idx, idx + b_size))
                    printf("[ERROR:Model:load] Failed to load %s layer. (idx:%d, w_size:%d)\n", layers[i]->getName(), idx, w_size);
                idx += b_size;
                this->layers[i]->load_params(W, b);
            }
        }
    };
}

#endif
/** End of moidel.hpp **/