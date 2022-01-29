#include <cstdio>
#include <string>

#define DEBUG_SSL

namespace custom_nn {
    namespace tensor {
        template <typename T>
        class Tensor {
        private:
            T* data;
            size_t size;
            int tensor_id;
        public:
            static int tensor_id_cnt;
            Tensor(size_t size) : size(size) {
                this->data = new T[size];
                tensor_id = tensor_id_cnt++;
            }
            ~Tensor() {
                delete[] this->data;
            }
            T operator[](int i){
                if (i > this->size || i < 0) {
                    fprintf(stderr, "[ERROR] Indexing out of range in %d tensor. (index: %d, size: %d)\n",
                        tensor_id, i, this->size);
                }
                return this->data[i];
            }
        };
    }

    namespace layer {
        template <typename T>
        class Layer {
        private:
            tensor::Tensor* p_weights;
            tensor::Tensor* p_bias;
        public:
            Layer((size_t w_size, size_t b_size)) {
                p_weights = new tensor::Tensor<T>(w_size);
                p_bias    = new tensor::Tensor<T>(b_size);
            }
            tensor::Tensor forward(tensor::Tensor p_x) {
                #ifdef DEBUG_SSL
                #endif
            }
            tensor::Tensor backward(tensor::Tensor p_dout) {

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
        public:

        };

        template <typename T>
        class ReLU : public Layer<T> {

        };

        template <typename T>
        class MaxPool2D : public Layer<T> {

        };
    }
}

