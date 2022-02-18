/**
 *
 * Deep Learning Seminar Task 2: Custom Neural Netweork Build with C++
 *
 * Written by Kwon, Han Kyul
 * File name: operator.h
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

#ifndef LAYER_H
#define LAYER_H

#include <stdio.h>
#include <string>
#include "common.hpp"
#include "tensor.hpp"

#define MAX_DIM 10

using namespace std;
using namespace tensor;

namespace layer {
    template <typename T>
    class Layer {
    private:
    protected:
        string layerName;
        Tensor<T>*  W = nullptr;    // Weight
        Tensor<T>* dW = nullptr;    // derivative-Weight
        Tensor<T>*  b = nullptr;    // bias
        Tensor<T>* db = nullptr;    // derivative-bias
    public:
        Layer(const char* cstr_layerName=nullptr) : layerName(cstr_layerName) {
        
        }
        ~Layer() {
            if (W)  delete W;
            if (dW) delete dW;
            if (b)  delete b;
            if (db) delete db;
        }
        virtual Tensor<T>* forward(Tensor<T>* x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:Layer] Forward\n");
            #endif
            return x;
        }
        virtual Tensor<T>* backward(Tensor<T>* dout) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:Layer] Backward\n");
            #endif
            return dout;
        }
        void load_params(Tensor<T> &W, Tensor<T> &b) {
            (*this->W) = W;
            (*this->b) = b;
        }
        int getWeightSize() {
            if (this->W != nullptr) return this->W->getSize();
            else return 0;
        }
        const int* getWeightShape() {
            if (this->W != nullptr) return this->W->getShape();
            else return nullptr;
        }
        int getBiasSize() {
            if (this->b != nullptr) return this->b->getSize();
            else return 0;
        }
        const int* getBiasShape() {
            if (this->b != nullptr) return this->b->getShape();
            else return nullptr;
        }
        int getWeightDim() {
            if (this->W != nullptr) return this->W->getDim();
            else return 0;
        }
        int getBiasDim() {
            if (this->b != nullptr) return this->b->getDim();
            else return 0;
        }
        const char* getName() {
            return this->layerName.c_str();
        }
    };


    template <typename T>
    class Conv2D : Layer<T> {
    private:
        const int in_ch, out_ch, k, p, s;
    public:
        Conv2D(const int in_ch, const int out_ch, const int k=3, const int p=1, const int s=1, const char* cstr_ln="") : in_ch(in_ch), out_ch(out_ch), k(k), p(p), s(s), Layer<T>(cstr_ln) {
            int shape_buf[MAX_DIM] = { 0, };
            shape_buf[0] = out_ch; shape_buf[1] = in_ch; shape_buf[2] = k; shape_buf[3] = k;
            this->W  = new Tensor<T>(4, shape_buf);  // W[out_ch][in_ch][k][k]
            this->dW = new Tensor<T>(4, shape_buf);
            shape_buf[0] = out_ch;
            this->b  = new Tensor<T>(1, shape_buf);  // b[out_ch];
            this->db = new Tensor<T>(1, shape_buf);
        }
        virtual Tensor<T>* forward(Tensor<T>* x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:Conv2D] Forward\n");
            #endif
            return x;
        }
        virtual Tensor<T>* backward(Tensor<T>* dout) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:Conv2D] Backward\n");
            #endif
            return dout;
        }
    };

    template <typename T>
    class FullyConnected : Layer<T> {
    private:
        const int in_fmap, out_fmap;
    public:
        FullyConnected(const int in_fmap, const int out_fmap, const char* cstr_ln) : in_fmap(in_fmap), out_fmap(out_fmap), Layer<T>(cstr_ln) {
            int shape_buf[MAX_DIM] = { 0, };
            shape_buf[0] = in_fmap; shape_buf[1] = out_fmap;
            this->W  = new Tensor<T>(2, shape_buf);   // W[in_fmap][out_fmap]
            this->dW = new Tensor<T>(2, shape_buf);   // 
            shape_buf[0] = out_fmap;
            this->b  = new Tensor<T>(1, shape_buf);
            this->db = new Tensor<T>(1, shape_buf);
        }
        virtual Tensor<T>* forward(Tensor<T>* x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:FullyConnected] Forward\n");
            #endif
            return x;
        }
        virtual Tensor<T>* backward(Tensor<T>* dout) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:FullyConnected] Backward\n");
            #endif
            return dout;
        }
    };


    template <typename T>
    class ReLU : Layer<T> {
    public:
        ReLU(const char* cstr_ln) : Layer<T>(cstr_ln) {

        }
        virtual Tensor<T>* forward(Tensor<T>* x) {
            Tensor<T>* ret = new Tensor<T>(x->getDim(), x->getShape());
            #ifdef LAYER_DEBUG
            printf("[DEBUG:ReLU] Forward\n");
            #endif
            const int* x_shape = x->getShape();
            for (int i = 0; i < x->getSize(); i++) {
                (*ret)[i] = (*x)[i] < 0 ? 0 : (*x)[i];
            }
            return ret;
        }
        virtual Tensor<T>* backward(Tensor<T>* dout) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:ReLU] Backward\n");
            #endif
            return dout;
        }
    };


    template <typename T>
    class MaxPool2D : Layer<T> {
    private:
        const int k, s; // k: kernel size, s: stride
    public:
        MaxPool2D(const int k=2, const int s=2, const char* cstr_ln="") : k(k), s(s), Layer<T>(cstr_ln) {

        }
        virtual Tensor<T>* forward(Tensor<T>* x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:MaxPool2D] Forward\n");
            #endif
            return x;
        }
        virtual Tensor<T>* backward(Tensor<T>* dout) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:MaxPool2D] Backward\n");
            #endif
            return dout;
        }
    };

}

#endif
 /** End of layer.hpp **/