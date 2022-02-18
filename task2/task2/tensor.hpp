/**
 *
 * Deep Learning Seminar Task 2: Custom Neural Netweork Build with C++
 *
 * Written by Kwon, Han Kyul
 * File name: tensor.h
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

#ifndef TENSOR_H
#define TENSOR_H

#include <cstdio>
#include <cassert>
#include <cstring>
#include "common.hpp"

namespace tensor {
    template <typename T>
    class Tensor {
    private:
        T* data = nullptr;
        int dim;
        int size = 1;
        int* shape = nullptr;
    public:
        // Default Constructor
        Tensor() {

        }

        // Copy Constructor
        Tensor(Tensor<T>& t) {
            if (data == nullptr) {
                delete data;
                data = nullptr;
            }
            if (shape == nullptr) {
                delete shape;
                shape = nullptr;
            }

            data = new T[t.size];
            assert(data);
            memcpy(data, t.data, t.size);

            shape = new T[t.dim];
            assert(shape);
            memcpy(shape, t.shape, t.dim);
        }

        // Constructor #1
        Tensor(int dim, const int shape[]) : dim(dim) {
            this->shape = new int[dim];
            assert(this->shape);
            for (int i = 0; i < dim; i++) {
                this->size *= shape[i];
                this->shape[i] = shape[i];
            }
            this->data = new T[size];
            assert(this->data);
            memset(this->data, 0, size * sizeof(T));
        }

        // Constructor #2
        Tensor(const T* data, int dim, int size, const int shape[]) : dim(dim) {
            this->data = new T[size];
            memcpy(this->data, data, size*sizeof(T));
        }

        // Destructor
        ~Tensor() {
            delete data;
            delete shape;
        }

        // Indexing Operator([]) Overloading
        T& operator[](int idx) {
            return data[idx];
        }
    };
}

#endif
/** End of tensor.hpp **/