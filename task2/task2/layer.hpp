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
        Tensor<T>* p_W  = nullptr;    // Weight
        Tensor<T>* p_dW = nullptr;    // derivative-Weight
        Tensor<T>* p_b  = nullptr;    // bias
        Tensor<T>* p_db = nullptr;    // derivative-bias
        Tensor<T>* p_x;     // Input from prev-layer
        Tensor<T>  y;       // Output of current-layer
        Tensor<T>* p_din;   // derviative-In
        Tensor<T>  dout;    // derviative-Out

        // Padding
        Tensor<T>* pad(Tensor<T> *p_x, const int p) {
            Tensor<T>& x = *p_x;
            const int* x_shape = x.getShape();
            const int B = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
            const int padded_shape[] = { B, C, H + 2 * p, W + 2 * p };
            Tensor<T> *p_padded_x = new Tensor<T>(4, padded_shape);
            Tensor<T>& padded_x = *p_padded_x;
            padded_x.dataInit();
            int padded_x_index[4] = { 0, };
            int x_index[4] = { 0, };
            for (int b = 0; b < x_shape[0]; b++) {
                padded_x_index[0] = b;
                x_index[0] = b;
                for (int c = 0; c < x_shape[1]; c++) {
                    padded_x_index[1] = c;
                    x_index[1] = c;
                    for (int h = 0; h < x_shape[2]; h++) {
                        padded_x_index[2] = h + p;
                        x_index[2] = h;
                        for (int w = 0; w < x_shape[3]; w++) {
                            padded_x_index[3] = w + p;
                            x_index[3] = w;
                            padded_x[index_calc(4, padded_shape, padded_x_index)] = x[index_calc(4, x_shape, x_index)];
                        }
                    }
                }
            }
            return p_padded_x;
        }
        
        // Image-to-Column
        Tensor<T>* im2col(Tensor<T>* p_im, const int k, const int s = 1, const int p = 0) {
            Tensor<T>& im = *p_im;
            const int* im_shape = im.getShape();
            const int B = im_shape[0], C = im_shape[1], H = im_shape[2], W = im_shape[3];
            const int out_h = (H + 2 * p - k) / s + 1;
            const int out_w = (W + 2 * p - k) / s + 1;

            Tensor<T>* p_padded_im = pad(p_im, p);
            Tensor<T>& padded_im = *p_padded_im;
            const int column_shape[] = { B, C, out_h, out_w, k, k };
            const int *padded_im_shape = padded_im.getShape();

            Tensor<T>* p_column = new Tensor<T>(6, column_shape);
            Tensor<T>& column = *p_column;
            //const int column_index[6] = { 0, };
            //const int padded_im_index[4] = { 0, };
            for (int b = 0; b < B; b++) {   // Batch
                for (int c = 0; c < C; c++) {   // Channel
                    for (int y = 0; y < k; y++) {   // Kernel Height
                        int y_max = y + s + out_h;
                        for (int x = 0; x < k; x++) {   // Kernel Width
                            int x_max = x + s * out_w;
                            for (int oh = 0; oh < out_h; oh++) {    // Output Height
                                for (int ow = 0; ow < out_w; ow++) {    // Output Width
                                    const int column_index[]    = { b, c, oh, ow, y, x };
                                    const int padded_im_index[] = { b, c, y + s * oh, x + s * ow };
                                    column[index_calc(6, column_shape, column_index)] = padded_im[index_calc(4, padded_im_shape, padded_im_index)];
                                }
                            }
                        }
                    }
                }
            }
            delete p_padded_im;
            return p_column;
        }
        
        // Column-to-Image
        Tensor<T>* col2im(Tensor<T>* p_col, const int k, const int s = 1, const int p = 0) {
            Tensor<T>& col = *p_col;
            const int* col_shape = col.getShape();
            const int B = col_shape[0], C = col_shape[1], H = col_shape[2], W = col_shape[3];
            int out_h = (H + 2 * p - k) / s + 1;
            int out_w = (W + 2 * p - k) / s + 1;
        }
    public:
        Layer(const char* cstr_layerName=nullptr) : layerName(cstr_layerName) {
        }
        ~Layer() {
            if (p_W)  delete p_W;
            if (p_dW) delete p_dW;
            if (p_b)  delete p_b;
            if (p_db) delete p_db;
        }
        virtual Tensor<T>* forward(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:Layer] Forward\n");
            #endif
            this->p_x = p_x;
            return &y;
        }
        virtual Tensor<T>* backward(Tensor<T>* p_din) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:Layer] Backward\n");
            #endif
            this->p_din = p_din;
            return &dout;
        }
        // Prepare Layer's output-tensor(y)
        virtual Tensor<T>* compile(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:LAYER] Compile\n");
            #endif
            this->p_x = p_x;
            y.reshape(*p_x);
            return &y;
        }
        void load_params(Tensor<T>* p_W, Tensor<T>* p_b) {
            this->p_W = p_W;
            this->p_b = p_b;
        }
        int getWeightSize() {
            if (this->p_W != nullptr) return this->p_W->getSize();
            else return 0;
        }
        const int* getWeightShape() {
            if (this->p_W != nullptr) return this->p_W->getShape();
            else return nullptr;
        }
        int getBiasSize() {
            if (this->p_b != nullptr) return this->p_b->getSize();
            else return 0;
        }
        const int* getBiasShape() {
            if (this->p_b != nullptr) return this->p_b->getShape();
            else return nullptr;
        }
        int getWeightDim() {
            if (this->p_W != nullptr) return this->p_W->getDim();
            else return 0;
        }
        int getBiasDim() {
            if (this->p_b != nullptr) return this->p_b->getDim();
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
            this->p_W  = new Tensor<T>(4, shape_buf);  // W[out_ch][in_ch][k][k]
            this->p_dW = new Tensor<T>(4, shape_buf);
            shape_buf[0] = out_ch;
            this->p_b  = new Tensor<T>(1, shape_buf);  // b[out_ch];
            this->p_db = new Tensor<T>(1, shape_buf);
        }
        virtual Tensor<T>* forward(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:Conv2D] Forward\n");
            #endif
            this->p_x = p_x;
            Tensor<T>& x = *p_x;
            const int  x_dim   = x.getDim();      // Dimension: 4
            const int* x_shape = x.getShape();  // B, C, H, W
            const int B = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
            int out_h = 1 + (int)( (H - this->k + 2 * this->p) / this->s );
            int out_w = 1 + (int)( (W - this->k + 2 * this->p) / this->s );
            Tensor<T>* p_col = this->im2col(p_x, this->k, this->s, this->p);    // col[B][in_ch][out_h][out_w][k][k]
            Tensor<T>& col = *p_col;
            this->y.dataInit();
            const int* y_shape = this->y.getShape();    // B, out_ch, out_h, out_w
            const int* col_shape = col.getShape();      // B, out_ch, out_h, out_w, k, k
            const int* W_shape = this->p_W->getShape(); // out_ch, in_ch, k, k
            int y_index[4] = { 0, };
            int col_index[6] = { 0, };
            int W_index[4] = { 0, };
            for (int b = 0; b < B; b++) {
                y_index[0]   = b;
                col_index[0] = b;
                for (int o_ch = 0; o_ch < this->out_ch; o_ch++) {
                    y_index[1] = o_ch;
                    W_index[0] = o_ch;
                    for (int o_h = 0; o_h < out_h; o_h++) {
                        y_index[2]   = o_h;
                        col_index[2] = o_h;
                        for (int o_w = 0; o_w < out_w; o_w++) {
                            y_index[3]   = o_w;
                            col_index[3] = o_w;
                            for (int i_ch = 0; i_ch < in_ch; i_ch++) {
                                col_index[1] = i_ch;
                                W_index[1]   = i_ch;
                                for (int k_y = 0; k_y < this->k; k_y++) {
                                    col_index[4] = k_y;
                                    W_index[2]   = k_y;
                                    for (int k_x = 0; k_x < this->k; k_x++) {
                                        col_index[5] = k_x;
                                        W_index[3]   = k_x;
                                        this->y[index_calc(4, y_shape, y_index)] += col[index_calc(6, col_shape, col_index)] * (*this->p_W)[index_calc(4, W_shape, W_index)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            //if (this->y.shapeCheck())
            delete p_col;
            return &(this->y);
        }
        virtual Tensor<T>* backward(Tensor<T>* p_din) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:Conv2D] Backward\n");
            #endif
            this->p_din = p_din;
            return &this->dout;
        }
        virtual Tensor<T>* compile(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:LAYER:Conv2D] Compile\n");
            #endif
            this->p_x = p_x;
            const int* x_shape = p_x->getShape();
            Tensor<T>& x = *p_x;
            const int B = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
            int out_h = 1 + ((H - this->k + 2 * this->p) / this->s);
            int out_w = 1 + ((W - this->k + 2 * this->p) / this->s);
            const int y_shape[] = { B, this->out_ch, out_h, out_w };
            this->y.reshape(4, y_shape);    // y[B][out_ch][out_h][out_w]
            this->dout.reshape(4, y_shape);
            return &this->y;
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
            this->p_W  = new Tensor<T>(2, shape_buf);     // W[in_fmap][out_fmap]
            this->p_dW = new Tensor<T>(2, shape_buf);     // dW[in_fmap][out_fmap]
            shape_buf[0] = out_fmap;
            this->p_b  = new Tensor<T>(1, shape_buf);     // b[out_fmap]
            this->p_db = new Tensor<T>(1, shape_buf);     // db[out_fmap]
        }
        virtual Tensor<T>* forward(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:FullyConnected] Forward\n");
            #endif
            this->y.dataInit();
            this->p_x = p_x;
            Tensor<T>& x = *this->p_x;
            Tensor<T>& W = *this->p_W;
            // Read in Flatten-Shape
            const int* x_shape = x.getShape();
            const int BatchSize = x_shape[0];
            for (int b = 0; b < BatchSize; b++) {
                for (int i = 0; i < in_fmap; i++) {
                    for (int o = 0; o < out_fmap; o++) {
                        this->y[b*out_fmap + o] += x[b*in_fmap + i] * W[i*out_fmap + o];
                    }
                }
            }
            return &this->y;
        }
        virtual Tensor<T>* backward(Tensor<T>* p_din) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:FullyConnected] Backward\n");
            #endif
            this->p_din = p_din;
            Tensor<T>& din = *p_din;
            return &this->dout;
        }
        virtual Tensor<T>* compile(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:LAYER:FullyConnected] Compile\n");
            #endif
            const int* x_shape = p_x->getShape();
            const int y_shape[] = { x_shape[0], this->out_fmap };
            this->y.reshape(2, y_shape);
            this->dout.reshape(2, y_shape);
            return &this->y;
        }
    };


    template <typename T>
    class ReLU : Layer<T> {
    public:
        ReLU(const char* cstr_ln) : Layer<T>(cstr_ln) {

        }
        virtual Tensor<T>* forward(Tensor<T>* p_x) {
            this->p_x = p_x;
            Tensor<T>& x = *p_x;
            const int* out_shape = x.getShape();
            #ifdef LAYER_DEBUG
            printf("[DEBUG:ReLU] Forward\n");
            #endif
            for (int i = 0; i < x.getSize(); i++) {
                this->y[i] = x[i] < 0 ? 0 : x[i];
            }
            return &this->y;
        }
        virtual Tensor<T>* backward(Tensor<T>* p_din) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:ReLU] Backward\n");
            #endif
            this->p_din = p_din;
            return &this->dout;
        }
        virtual Tensor<T>* compile(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:LAYER:ReLU] Compile\n");
            #endif
            this->y.reshape(*p_x);
            return &this->y;
        }
    };


    template <typename T>
    class MaxPool2D : Layer<T> {
    private:
        const int k, s; // k: kernel size, s: stride
    public:
        MaxPool2D(const int k=2, const int s=2, const char* cstr_ln="") : k(k), s(s), Layer<T>(cstr_ln) {

        }
        virtual Tensor<T>* forward(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:MaxPool2D] Forward\n");
            #endif
            this->p_x = p_x;
            Tensor<T>& x = *p_x;
            const int  x_dim = x.getDim();      // Dimension: 4
            const int* x_shape = x.getShape();  // B, C, H, W
            const int B = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
            int out_h = 1 + (int)((H - this->k) / this->s);
            int out_w = 1 + (int)((W - this->k) / this->s);
            Tensor<T>* p_col = this->im2col(p_x, this->k, this->s);    // col[B][C][out_h][out_w][k][k]
            Tensor<T>& col = *p_col;
            const int* col_shape = col.getShape();
            const int out_shape[] = { B, C, out_h, out_w };
            int col_index[6] = { 0, };
            int out_index[4] = { 0, };
            for (int b = 0; b < B; b++) {
                col_index[0] = b;
                out_index[0] = b;
                for (int c = 0; c < C; c++) {
                    col_index[1] = c;
                    out_index[1] = c;
                    for (int o_h = 0; o_h < out_h; o_h++) {
                        col_index[2] = o_h;
                        for (int o_w = 0; o_w < out_w; o_w++) {
                            col_index[3] = o_w;
                            T max_val = -INFINITY;
                            for (int k_y = 0; k_y < k; k_y++) {
                                out_index[2] = k_y;
                                for (int k_x = 0; k_x < k; k_x++) {
                                    out_index[3] = k_x;
                                    max_val = max_val < col[index_calc(6, col_shape, col_index)] ? col[index_calc(6, col_shape, col_index)] : max_val;
                                }
                            }
                            this->y[index_calc(4, out_shape, out_index)] = max_val;
                        }
                    }
                }
            }
            return &this->y;
        }
        virtual Tensor<T>* backward(Tensor<T>& din) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:MaxPool2D] Backward\n");
            #endif
            this->p_din = &din;
            return &this->dout;
        }
        virtual Tensor<T>* compile(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:LAYER:MaxPool2D] Compile\n");
            #endif
            const int* x_shape = p_x->getShape();
            const int B = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
            int out_h = 1 + (int)((H - this->k) / this->s);
            int out_w = 1 + (int)((W - this->k) / this->s);
            const int y_shape[] = { B, C, out_h, out_w };
            this->y.reshape(4, y_shape);
            return &this->y;
        }
    };

}

#endif
 /** End of layer.hpp **/