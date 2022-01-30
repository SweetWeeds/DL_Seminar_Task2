#ifndef _CUSTOM_NN_H
#define _CUSTOM_NN_H

#include <cstdio>
#include <string>

namespace custom_nn {
    namespace tensor {
        int TENSOR_ID_CNT = 0;

        /**
         * Tensor
         */
        template <typename T>
        class Tensor {
        private:
            T* data = nullptr;
            size_t size = 0;
            const int tensor_id;
        public:
            // Constructor
            Tensor(size_t size=0) : size(size), tensor_id(TENSOR_ID_CNT++) {
                if (size > 0) {
                    this->data = new T[size];
                    memset(data, 0, size);
                }
                else {
                    this->data = nullptr;
                }
            }

            // Copy Constructor
            Tensor(const Tensor& t) {
                delete[] this->data;
                this->data = new T[t.getSize()];
                for (int i = 0; i < t.getSize(); i++) {
                    this->data[i] = t[i];
                }
            }

            // Destructor
            ~Tensor() {
                delete[] this->data;
            }

            // Indexing Operator Overloading
            T& operator[](size_t i){
                if (i > this->size || i < 0) {
                    fprintf(stderr, "[ERROR] Indexing out of range in %d tensor. (index: %d, size: %d)\n",
                        tensor_id, i, this->size);
                    i = 0;
                }
                return this->data[i];
            }

            size_t getSize(void) {
                #ifdef DEBUG_TENSOR
                printf("[INFO] Tensor %d's size: %d\n", this->tensor_id, this->size);
                #endif
                return size;
            }

            int getID(void) {
                return this->tensor_id;
            }

            void resize(size_t new_size) {
                delete[] this->data;
                this->data = new T[new_size];
                this->size = new_size;
            }
        };
    }

    namespace helper {
        template <typename T>
        tensor::Tensor<T> im2col(tensor::Tensor<T> input_data,
            const int Batch, const int Channel, const int Height, const int Width,
            const int filter_size, const int stride=1, const int pad=0) {
            /*
            const OUT_Height = (Height + 2 * pad - filter_size) / stride + 1;
            const OUT_Width = (Width + 2 * pad - filter_size) / stride + 1;
            tensor::Tensor<T> img();
            */
        }
    }

    namespace layer {
        /**
         * Layer
         */
        int LAYER_ID_CNT = 0;
        template <typename T>
        class Layer {
        private:
            tensor::Tensor<T> weights;      // Weight
            tensor::Tensor<T> bias;         // Bias
            tensor::Tensor<T> d_weights;    // Weight Derivative
            tensor::Tensor<T> d_bias;       // Bias Derivative
            int layer_id;
        public:
            Layer(size_t w_size=0, size_t b_size=0) : layer_id(LAYER_ID_CNT++) {
                weights.resize(w_size);
                d_weights.resize(w_size);
                bias.resize(b_size);
                d_bias.resize(b_size);
            }
            Layer(tensor::Tensor<T> weights, tensor::Tensor<T> bias) {
                this->weights = weights;
                this->bias    = bias;
            }
            // Forward
            tensor::Tensor<T> forward() {
                #ifdef DEBUG_LAYER
                printf("[INFO:Layer %02d] Forward\n", this->layer_id);
                #endif
            }
            /**
             * Back Propagation
             * Return: dout
             */
            tensor::Tensor<T> backward() {
                #ifdef DEBUG_LAYER
                printf("[INFO:Layer %02d] Backward\n", this->layer_id);
                #endif
            }
            /**
             * Weight Update
             *  Update the weights with learning rate.
             *  T lr: Learning Rate
             */
            void update(T lr) {
                #ifdef DEBUG_LAYER
                printf("[INFO:Layer %02d] Update\n", this->layer_id);
                #endif
                for (int i = 0; i < weights.getSize(); i++)
                    weights[i] += lr * d_weights[i];
                for (int i = 0; i < bias.getSize(); i++)
                    bias[i]    += lr * d_bias[i];

            }
        };

        template <typename T>
        class Conv2D : public Layer<T> {
        private:

        public:

        };

        template <typename T>
        class Affine : public Layer<T> {
        private:
            const int InputSize, OutputSize;
        public:
            Affine(const int InputSize, const int OutputSize) : InputSize(InputSize), OutputSize(OutputSize),
                Layer(w_size=InputSize*OutputSize, b_size=OutputSize) {
                
            }
            tensor::Tensor<T> forward(tensor::Tensor<T> x,
                const int Batch, const int InputSize) {
                tensor::Tensor<T> ret_T(Batch*OutputSize);
                for (int b = 0; b < Batch; b++) {
                    for (int i = 0; i < InputSize; i++) {

                    }
                }
            }
        };

        template <typename T>
        class ReLU : public Layer<T> {
        private:

        public:
            tensor::Tensor<T> forward(tensor::Tensor<T> x) {
                size_t x_size = x.getSize();
                tensor::Tensor<T> ret_T(x_size);
                for (size_t i = 0; i < x_size; i++) {
                    ret_T[i] = x[i] < 0 ? 0 : x[i];
                }
                return ret_T;
            }

            tensor::Tensor<T> backward(tensor::Tensor<T> d_out) {

            }
        };

        template <typename T>
        class MaxPool2D : public Layer<T> {
        private:
        public:

        };
    }
}

#endif