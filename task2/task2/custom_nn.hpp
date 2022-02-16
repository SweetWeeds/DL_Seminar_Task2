#ifndef _CUSTOM_NN_H
#define _CUSTOM_NN_H

#include <cstdio>
#include <cstring>
#include <cmath>
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
            int dim = 0;
            int *shape = nullptr;
            int size = 1;
            const int tensor_id;
        public:
            // Constructor
            Tensor(int dim=0, int *shape=nullptr) : dim(dim), tensor_id(TENSOR_ID_CNT++) {
                if (dim == 0) return;
                this->shape = new int[dim];
                for (int i = 0; i < dim; i++) {
                    this->size *= shape[i];
                    this->shape[i] = shape[i];
                }
                this->data = new T[this->size];
                memset(this->data, 0, this->size * sizeof(T));
            }

            // Copy Constructor
            Tensor(const Tensor& t) : tensor_id(TENSOR_ID_CNT++) {
                // Resize Tensor Shape
                this->resize(t.getDim, t.getShape);

                // Copy Data
                this->data = new T[size];
                for (int i = 0; i < size; i++) {
                    this->data[i] = t[i];
                }
            }

            // Destructor
            ~Tensor() {
                delete[] this->data;
            }

            // Indexing Operator Overloading
            T& operator[](int i){
                if (i > this->size || i < 0) {
                    fprintf(stderr, "[ERROR] Indexing out of range in %d tensor. (index: %d, size: %d)\n",
                        tensor_id, i, this->size);
                    i = 0;
                }
                return this->data[i];
            }

            int getDim(void) {
                return dim;
            }

            const int* getShape(void) {
                return shape;
            }

            int getSize(void) {
                #ifdef DEBUG_TENSOR
                printf("[INFO] Tensor %d's size: %d\n", this->tensor_id, this->size);
                #endif
                return size;
            }

            int getID(void) {
                #ifdef DEBUG_TENSOR
                printf("[INFO] Tensor ID: %06d\n", this->tensor_id);
                #endif
                return this->tensor_id;
            }

            void resize(int dim, const int* shape) {
                // Delete Allocated Memory
                delete[] this->shape;

                // Get dimension and shape of Tensor "t"
                this->dim = dim;
                this->shape = new int[dim];
                int tmp_size = 1;
                for (int i = 0; i < dim; i++) {
                    tmp_size *= shape[i];
                    this->shape[i] = shape[i];
                }

                // If size is not matching, delete memory and allocate new memory space.
                if (tmp_size != this->size) {
                    delete[] this->data;
                    this->size = tmp_size;
                    this->data = new T[this->size];
                }
            }


            bool shapeCheck(int dim1, int* shape1, int dim2, int* shape2) {
                if (dim1 != dim2) return false;
                for (int i = 0; i < dim1; i++) {
                    if (shape1[i] != shape2[i]) return false;
                }
                return true;
            }
        };
    }

    namespace helper {
        using namespace tensor;

        /**
         * Cross Entropy Error
         *  Input
         *      Tensor<T> y[B, N]: Prediction Value
         *      Tensor<T> t[N]: Label
         *  Output
         *      T ret : -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
         */
        template <typename T>
        T CrossEntropyError(Tensor<T> y, Tensor<int> t) {
            T ret = 0;
            const int* y_shape = y.getShape();
            const int* t_shape = t.getShape();
            const int BatchSize = y_shape[0];

            for (int b = 0; b < BatchSize; b++) {
                for (int i = 0; i < y_shape[1]; i++) {
                    if (t[i] == 1) {
                        ret -= log(y[b*y_shape[1]+i]);
                    }
                }
            }
            ret /= BatchSize;
            return ret;
        }


        /**
         * Cross Entropy Error
         *  Input
         *      Tensor<T> x[B, N]: Input Tensor
         *  Output
         *      Tensor<T> ret : np.exp(x) / np.sum(np.exp(x))
         */
        template <typename T>
        Tensor<T> Softmax(Tensor<T> x) {
            Tensor<T> ret;
            const int* x_shape = x.getShape();
            const int BatchSize = x_shape[0];
            ret.resize(x.getDim(), x_shape);
            for (int b = 0; b < BatchSize; b++) {
                T sum = 0, max = -INFINITY;
                for (int i = 0; i < x_shape[1]; i++) {
                    ret[b*x_shape[1] + i] = exp(x[b*x_shape[1] + i]);
                    max = max < ret[b*x_shape[1] + i] ? ret[b*x_shape[1] + i] : max;
                }
                for (int i = 0; i < x_shape[1]; i++) {
                    ret[b*x_shape[1] + i] -= max;
                    sum += ret[b*x_shape[1] + i];
                }
                for (int i = 0; i < x_shape[1]; i++) {
                    ret[b*x_shape[1] + i] /= sum;
                }
            }
            return ret;
        }


        template <typename T>
        Tensor<T> im2col(Tensor<T> ifmap, const int filter_size, const int stride=1, const int pad=0) {
            const int* ifmap_shape = ifmap.getShape();
            const int B = ifmap_shape[0], C = ifmap_shape[1],
                      H = ifmap_shape[2], W = ifmap_shape[3];

            const int out_H = (H + 2 * pad - filter_size) / stride + 1;
            const int ouw_W = (W + 2 * pad - filter_size) / stride + 1;

            // Padding
            const int pH = H + 2 * pad, pW = W + 2 * pad;
            const int pifmap_size[] = { B, C, H+2*pad, W+2*pad };
            Tensor<T> pifmap(4, pifmap_size);
            for (int b = 0; b < B; b++) {
                for (int c = 0; c < C; c++) {
                    for (int h = pad; h < pad + H; h++) {
                        for (int w = pad; w < pad + W; w++) {
                            pifmap[b*(C*pH*pW)+c*(pH*pW)+h*pH+w] = pifmap[b*C*H*W+c*H*W+h*W+w];
                        }
                    }
                }
            }

            // Image to Column
            for (int i = 0; i < filter_size; i++) {
                for (int j = 0; j < filter_size; j++) {

                }
            }
        }

        template <typename T>
        Tensor<T> col2im(Tensor<T> col, const int* ifmap_shape, const int filter_size, const int stride = 1, const int pad = 0) {
            const int B = ifmap_shape[0], C = ifmap_shape[1], H = ifmap_shape[2], W = ifmap_shape[3];
            const int out_H = (H + 2 * pad - filter_size) / stride + 1;
            const int out_W = (W + 2 * pad - filter_size) / stride + 1;

        }
    }

    namespace layer {
        using namespace tensor;
        /**
         * Layer
         */
        int LAYER_ID_CNT = 0;
        std::string layerName;
        template <typename T>
        class Layer {
        private:
            int layer_id;
        public:
            Layer(std::string layerName) : layerName(layerName), layer_id(LAYER_ID_CNT++) {
            }


            // Forward
            virtual Tensor<T> forward(Tensor<T> x) {
                #ifdef DEBUG_LAYER
                printf("[INFO:%s] Forward\n", layerName.c_str());
                #endif
            }


            /**
             * Back Propagation
             * Return: dout
             */
            virtual Tensor<T> backward(Tensor<T> dout) {
                #ifdef DEBUG_LAYER
                printf("[INFO:%s] Backward\n", layerName.c_str());
                #endif
            }


            /**
             * Weight Update
             *  Update the weights with learning rate.
             *  T lr: Learning Rate
             */
            virtual void update(T lr) {
                #ifdef DEBUG_LAYER
                printf("[INFO:%s] Update\n", layerName.c_str());
                #endif
            }
        };

        template <typename T>
        class Conv2D : public Layer<T> {
        private:
            const int in_channels, out_channels, kernel_size, padding, stride;
            const int* x_shape;
            int weight_shape[4] = { out_channels, in_channels, kernel_size, kernel_size };
            int bias_shape[1] = { out_channels };

            Tensor<T>  W(4, weight_shape);
            Tensor<T> dW(4, weight_shape);
            Tensor<T>  b(1, out_channels);
            Tensor<T> db(1, out_channels);
        public:
            Conv2D(const int in_channels, const int out_channels, const int kernel_size, const int padding, const int stride, const char* layerName) {

            }
            Tensor<T> forward(Tensor<T> x) {
                x_shape = x.getShape();
                const int B = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
                const int out_H = 1 + (int)((H - this->kernel_size + 2 * this->padding) / this->stride);
                const int out_W = 1 + (int)((W - this->kernel_size + 2 * this->padding) / this->stride);

                Tensor<T> col = helper::im2col(x, this->kernel_size, this->stride, this->padding);

            }
        };

        template <typename T>
        class MaxPool2D : public Layer<T> {
        private:
        public:

        };

        template <typename T>
        class Affine : public Layer<T> {
        private:
            const int InputSize, OutputSize;
        public:
            Affine(const int InputSize, const int OutputSize) : InputSize(InputSize), OutputSize(OutputSize),
                Layer(InputSize*OutputSize, OutputSize) {
                
            }
            Tensor<T> forward(Tensor<T> x) {
                Tensor<T> ret_T(Batch*OutputSize);
                for (int b = 0; b < Batch; b++) {
                    for (int i = 0; i < InputSize; i++) {

                    }
                }
            }
        };

        template <typename T>
        class ReLU : public Layer<T> {
        private:
            bool* mask = nullptr;
        public:
            ReLU(const int dim, ) {

            }
            Tensor<T> forward(Tensor<T> x) {
                int x_size = x.getSize();
                tensor::Tensor<T> ret_T(x_size);
                for (int i = 0; i < x_size; i++) {
                    ret_T[i] = x[i] < 0 ? 0 : x[i];
                }
                return ret_T;
            }

            Tensor<T> backward(Tensor<T> d_out) {

            }
        };

    }
}

#endif