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
 *      2. Modified by Han Kyul Kwon on Feb 17, 2022
 *          (a) Add codes for normal execution. (on parctice hours)
 *
 * Compiler used: MSVC++ 14.16 (Visual Studio 2017 version 15.9)
 *
 */
#ifndef MODEL_H
#define MODEL_H

#include <string.h>
#include <assert.h>
#include "common.hpp"
#include "data_loader.hpp"
#include "tensor.hpp"
#include "layer.hpp"

namespace model {
    using namespace tensor;
    using namespace layer;
    using namespace data_loader;

    template <typename T>
    class Model {
    private:
        Layer<T>* layers[MAX_LAYER_NUM];
        Criterion<T>* criterion;
        int num_layers = 0;
        bool is_compiled = false;
        char* modelName = nullptr;
        DataLoader<T> dl;
    public:
        Model(Layer<T>* layers[], Criterion<T>* criterion, int num_layers, const char* modelName) : num_layers(num_layers) {
            memcpy(this->layers, layers, num_layers * sizeof(Layer<T>*));
            int modelNameLen = strlen(modelName);
            this->modelName = new char[modelNameLen];
            memcpy(this->modelName, modelName, modelNameLen * sizeof(char));
        }

        ~Model() {
            for (int i = 0; i < num_layers; i++) {
                delete this->layers[i];
            }
            delete criterion;
        }

        Tensor<T>* forward(Tensor<T>* p_x) {
            if (!this->is_compiled) {
                printf("[WARNING:Model:forward] %s is not compiled!\n", this->modelName);
                return nullptr;
            }
            for (int i = 0; i < num_layers; i++) {
                p_x = this->layers[i]->forward(p_x);
            }
            return p_x;
        }

        int diff(Tensor<T>* p_x, const char* fileName) {
            if (!this->is_compiled) {
                printf("[WARNING:Model:diff] %s is not compiled!\n", this->modelName);
                return 1;
            }
            T maxDiff = 0;
            DataLoader<float> dl;
            dl.load_data(fileName);
            int idx = 0;
            // Forward
            printf("[INFO] Starting forward compare...\n");
            for (int i = 0; i < num_layers; i++) {
                p_x = this->layers[i]->forward(p_x);
                Tensor<T> golden_tensor(p_x->getDim(), p_x->getShape());
                T* p_data = golden_tensor.getData();
                dl.copyDataOfRange(p_data, idx, idx + p_x->getSize());
                T tmp = golden_tensor.getMaxDiff(*p_x);
                maxDiff = maxDiff < tmp ? tmp : maxDiff;
                printf("[INFO] %s layer's max difference:\t %e\n", this->layers[i]->getName(), maxDiff);
                idx += p_x->getSize();
            }
            // Calculate Loss
            //p_x = this->criterion->forward(p_x, );
            // Backward
            printf("[INFO] Starting backward compare...\n");
            for (int i = num_layers - 1; i >= 0; i--) {

            }
            return 0;
        }

        Tensor<T>* backward(Tensor<T>* dout) {
            if (!this->is_compiled) {
                printf("[WARNING:Model:backward] %s is not compiled!\n", this->modelName);
                return nullptr;
            }
            for (int i = num_layers - 1; i >= 0; i--) {
                dout = this->layers[i]->backward(dout);
            }
            return dout;
        }

        // Compile Model: Allocate each layer's output-tensor
        int compile(Tensor<T>* p_x) {
            for (int i = 0; i < num_layers; i++) {
                p_x = this->layers[i]->compile(p_x);
            }
            this->is_compiled = true;
            return 0;
        }

        int save(const char* fn) {
            // TODO
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
            int w_size, b_size, w_dim, b_dim;
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
                Tensor<T>* p_W = new Tensor<T>(w_dim, w_shape);
                Tensor<T>* p_b = new Tensor<T>(b_dim, b_shape);
                if (dl.copyDataOfRange(p_W->getData(), idx, idx + w_size))
                    printf("[ERROR:Model:load] Failed to load %s layer. (idx:%d, w_size:%d)\n", layers[i]->getName(), idx, w_size);
                idx += w_size;
                if (dl.copyDataOfRange(p_b->getData(), idx, idx + b_size))
                    printf("[ERROR:Model:load] Failed to load %s layer. (idx:%d, w_size:%d)\n", layers[i]->getName(), idx, w_size);
                idx += b_size;
                this->layers[i]->load_params(p_W, p_b);
            }
            return 0;
        }
    };
}

#endif
/** End of moidel.hpp **/