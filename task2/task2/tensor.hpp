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

        // Shape Checking Method (0: Normal, 1: Dimension mismatch, 2: Shape mistmatch)
        int shape_check(int dim1, const int* shape1, int dim2, const int* shape2) {
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
            int shape_chk = shape_check(this->dim, this->shape, t.getDim(), t.getShape());
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
                    this->dim = t.getDim();
                    delete this->shape;
                    this->shape = new int[this->dim];
                    assert(this->shape);
                }
                // Copy Shape
                memcpy(this->shape, t.getShape(), t.getDim()*sizeof(int));
            }
            memcpy(data, t.getData(), t.getSize()*sizeof(T));
        }

        // Constructor #1
        Tensor(int dim, const int shape[]) : dim(dim) {
            #ifdef TENSOR_DEBUG
            printf("[DEBUG:Tensor] Constructor #1\n");
            #endif
            if (this->shape) delete this->shape;
            this->shape = new int[dim];
            assert(this->shape);
            for (int i = 0; i < dim; i++) {
                this->size *= shape[i];
                this->shape[i] = shape[i];
            }
            if (this->data) delete this->data;
            this->data = new T[size];
            assert(this->data);
            memset(this->data, 0, size * sizeof(T));
        }

        // Constructor #2
        Tensor(const T* data, int dim, int size, const int shape[]) : dim(dim) {
            #ifdef TENSOR_DEBUG
            printf("[DEBUG:Tensor] Constructor #2\n");
            #endif
            this->size = size;
            this->data = new T[size];
            memcpy(this->data, data, size*sizeof(T));
            this->shape = new int[dim];
            memcpy(this->shape, shape, dim * sizeof(int));
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
            #ifdef TENSOR_DEBUG
            if (idx > this->size || idx < 0) {
                printf("[WARNING:Tensor:operator[]] Indexing Out of Range (size:%d, idx:%d)\n", this->size, idx);
                return data[ idx < 0 ? 0 : (this->size-1) ];    // Return data[0] or data[size-1]
            }
            #endif
            return data[idx];
        }

        // Reshape
        //  0:  Succeeed Reshaping
        //  1:  Failed Reshaping
        int reshape(int dim, const int shape[]) {
            int ret = 0;
            int new_size = 1;
            for (int i = 0; i < dim; i++) {
                new_size *= shape[i];
            }
            if (new_size != this->size) {   // Size is not matching
                #ifdef TENSOR_DEBUG
                printf("[DEBUG:Tensor:reshape] Reshaping size is not matching.\n");
                #endif
                // Reallocate Data
                if (data) delete data;
                this->size = new_size;
                data = new T[new_size];
                assert(data);
                memset(data, 0, new_size * sizeof(T));

                // Reallocate Shape
                ret = 1;
            }
            this->dim = dim;
            delete this->shape;
            this->shape = new int[dim];
            assert(this->shape);
            memcpy(this->shape, shape, dim * sizeof(int));
            return ret;
        }

        int reshape(Tensor<T>& t) {
            this->reshape(t.getDim(), t.getShape());
            return 0;
        }
        

        T* getData() {
            return this->data;
        }

        int getSize() {
            return this->size;
        }

        int getDim() {
            return this->dim;
        }

        const int* getShape() {
            return this->shape;
        }

        // Shape Checking Method (0: Normal, 1: Dimension mismatch, 2: Shape mistmatch, 3: Tensor is not initialized)
        int shapeCheck(Tensor<T>& t) {
            if (this->data == nullptr || t.getData() == nullptr) return 3;
            return this->shape_check(this->dim, this->shape, t.getDim(), t.getShape());
        }

        int shapeCheck(const int dim, const int* shape) {
            if (this->data == nullptr) return 3;
            return this->shape_check(this->dim, this->shape, dim, shape);
        }

        void dataInit() {
            memset(this->data, 0, size * sizeof(T));
        }

        T getDiffError(Tensor<T>& t) {
            T maxDiff = 0;
            float error_rate = 0;
            int idx = 0;
            float tmp;
            if (t.getSize() != this->size) {
                #ifdef DEBUG_TENSOR
                printf("[DEBUG:Tensor:getMaxDiff] Size is not matching.\n");
                return INFINITY;
                #endif
            }
            for (int i = 0; i < this->size; i++) {
                tmp = this->data[i] - t[i];
                tmp = tmp < 0 ? -tmp : tmp;
                if (maxDiff < tmp) {
                    maxDiff = tmp;
                    error_rate = maxDiff / this->data[i] * 100.0f;
                    idx = i;
                }
            }
            return error_rate;
        }
    };
    inline int index_calc(const int dim, const int* shape, const int* index) {
        int ret = 0;
        for (int d = 0; d < dim; d++) {
            int tmp = index[d];
            for (int i = d + 1; i < dim; i++) {
                tmp *= shape[i];
            }
            ret += tmp;
        }
        return ret;
    }
}

#endif
/** End of tensor.hpp **/