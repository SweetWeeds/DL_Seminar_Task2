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

        // Shape Checking Method
        int shape_check(const int* shape1, int dim1, const int* shape2, int dim2) {
            if (dim1 != dim2) return 1;
            for (int i = 0; i < dim1; i++) {
                if (shape1[i] != shape2[i]) return 2;
            }
            return 0;
        }
    public:
        // Default Constructor
        Tensor() {

        }

        // Copy Constructor
        Tensor(Tensor<T>& t) {
            int shape_chk = shape_check(this->shape, this->dim, t.getShape(), t.getDim());
            if (shape_chk) {
                #ifdef TENSOR_DEBUG
                printf("[DEBUG:Tensor:Copy_Constructor] Shape is not matching.\n");
                #endif
                if (this->size != t.getSize()) {
                    #ifdef TENSOR_DEBUG
                    printf("[DEBUG:Tensor:Copy_Constructor] Size is not matching.(size1: %d, size2: %d)\n", this->size, t.getSize());
                    #endif
                    // Reallocate Data
                    delete data;
                    data = new T[t.getSize()];
                    assert(data);

                    // Reallocate Shape
                    delete shape;
                    shape = new int[t.getDim()];
                    assert(shape);
                }
                // Copy Shape
                memcpy(shape, t.getShape(), t.getDim());
            }
            memcpy(data, t.getData(), t.getSize());
        }

        // Constructor #1
        Tensor(int dim, const int shape[]) : dim(dim) {
            #ifdef TENSOR_DEBUG
            printf("[DEBUG:Tensor] Constructor #1\n");
            #endif
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
            #ifdef TENSOR_DEBUG
            printf("[DEBUG:Tensor] Constructor #2\n");
            #endif
            this->data = new T[size];
            memcpy(this->data, data, size*sizeof(T));
        }

        // Destructor
        ~Tensor() {
            #ifdef TENSOR_DEBUG
            printf("[DEBUG:Tensor] Destructor\n");
            #endif
            delete data;
            delete shape;
        }

        // Indexing Operator([]) Overloading
        T& operator[](int idx) {
            if (idx > this->size || idx < 0) {
                #ifdef TENSOR_DEBUG
                printf("[DEBUG:Tensor:operator[]] Indexing Out of Range\n");
                #endif
                return data[ idx < 0 ? 0 : (size-1) ];    // Return data[0] or data[size-1]
            }
            return data[idx];
        }

        // Reshape
        //  True:  Succeeed Reshaping
        //  False: Failed Reshaping
        bool reshape(int dim, const int shape[]) {
            int new_size = 1;
            for (int i = 0; i < dim; i++) {
                new_size *= shape[i];
            }
            if (new_size != this->size) {
                #ifdef TENSOR_DEBUG
                printf("[DEBUG:Tensor:reshape] Reshaping size is not matching.\n");
                #endif
                return false;
            }
            delete this->shape;
            this->shape = new int[dim];
            memcpy(this->shape, shape, dim * sizeof(int));
            return true;
        }

        T* getData() {
            return this->data;
        }

        int getSize() {
            return size;
        }

        int getDim() {
            return dim;
        }

        const int* getShape() {
            return this->shape;
        }
    };
}

#endif
/** End of tensor.hpp **/