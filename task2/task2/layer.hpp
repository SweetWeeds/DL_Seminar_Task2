/**
 *
 * Deep Learning Seminar Task 2: Custom Neural Netweork Build with C++
 *
 * Written by Kwon, Han Kyul
 * File name: operator.h
 * Program date: 2022.02.17.
 *
 * Written on Feb 17, 2022
 * Modified on Feb 21, 2022
 * Modification History:
 *      1. Written by Hankyul Kwon
 *      2. Modified by Han Kyul Kwon on Feb 17, 2022
 *          (a) Designed top layout of classes.
 *      3. Modified by Han Kyul Kwon on Feb 20, 2022
 *          (a) Conv2D, FullyConnected forward method complete.
 *      4. Modified by Han Kyul Kwon on Feb 21, 2022
 *          (a) Add SoftmaxWithLoss class.
 *
 * Compiler used: MSVC++ 14.16 (Visual Studio 2017 version 15.9)
 *
 */

#ifndef LAYER_H
#define LAYER_H

#include <stdio.h>
#include <string>
#include <math.h>
#include "common.hpp"
#include "tensor.hpp"

namespace layer {
    using namespace std;
    using namespace tensor;

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
        Tensor<T>* p_dout;   // derviative-In
        Tensor<T>  din;    // derviative-Out

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
            int column_index[6] = { 0, };
            int padded_im_index[4] = { 0, };
            for (int b = 0; b < B; b++) {   // Batch
                column_index[0] = b;
                padded_im_index[0] = b;
                for (int c = 0; c < C; c++) {   // Channel
                    column_index[1] = c;
                    padded_im_index[1] = c;
                    for (int y = 0; y < k; y++) {   // Kernel Height
                        int y_max = y + s + out_h;
                        column_index[4] = y;
                        for (int x = 0; x < k; x++) {   // Kernel Width
                            int x_max = x + s * out_w;
                            column_index[5] = x;
                            for (int oh = 0; oh < out_h; oh++) {    // Output Height
                                column_index[2] = oh;
                                padded_im_index[2] = y + s * oh;
                                for (int ow = 0; ow < out_w; ow++) {    // Output Width
                                    column_index[3] = ow;
                                    padded_im_index[3] = x + s * ow;
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
            const int out_h = (H + 2 * p - k) / s + 1;
            const int out_w = (W + 2 * p - k) / s + 1;

            const int padded_im_shape[] = { B, C, H + 2 * p + s - 1, W + 2 * p + s - 1 };
            const int im_H = H + s - 1, im_W = W + s - 1;
            const int im_shape[] = { B, C, im_H, im_W };
            //const int col_shape[] = { B, C, out_h, out_w, k, k };
            Tensor<T>* p_padded_im = new Tensor<T>(4, padded_im_shape);
            Tensor<T>* p_im = new Tensor<T>(4, im_shape);
            Tensor<T>& padded_im = *p_padded_im;
            Tensor<T>& im = *p_im;
            padded_im.dataInit();

            int padded_im_index[4] = { 0, };
            int col_index[6] = { 0, };
            int im_index[4] = { 0, };
            for (int b = 0; b < B; b++) {
                padded_im_index[0] = b;
                im_index[0] = b;
                col_index[0] = b;
                for (int c = 0; c < C; c++) {
                    padded_im_index[1] = c;
                    im_index[1] = c;
                    col_index[1] = c;
                    for (int y = 0; y < k; y++) {
                        int y_max = y + s * out_h;
                        col_index[4] = y;
                        for (int x = 0; x < k; x++) {
                            int x_max = x + s * out_w;
                            col_index[5] = x;
                            for (int o_h = 0; o_h < out_h; o_h++) {
                                padded_im_index[2] = y + s * o_h;
                                col_index[2] = o_h;
                                for (int o_w = 0; o_w < out_w; o_w++) {
                                    padded_im_index[3] = x + s * o_w;
                                    col_index[3] = o_w;
                                    padded_im[index_calc(4, padded_im_shape, padded_im_index)] += col[index_calc(6, col_shape, col_index)];
                                }
                            }
                        }
                    }
                    // Parse im from padded_im
                    for (int h = 0; h < im_H; h++) {
                        im_index[2] = h;
                        padded_im_index[2] = h + p;
                        for (int w = 0; w < im_W; w++) {
                            im_index[3] = w;
                            padded_im_index[3] = w + p;
                            im[index_calc(4, im_shape, im_index)] = padded_im[index_calc(4, padded_im_shape, padded_im_index)];
                        }
                    }
                }
            }
            delete p_padded_im;
            return p_im;
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
        virtual Tensor<T>* backward(Tensor<T>* p_dout) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:Layer] Backward\n");
            #endif
            this->p_dout = p_dout;
            return &din;
        }
        virtual void update(float lr=0.001) {
            const int W_size = p_W->getSize();
            const int b_size = p_b->getSize();
            for (int i = 0; i < W_size; i++) {
                (*p_W)[i] -= lr * (*p_dW)[i];
            }
            for (int i = 0; i < b_size; i++) {
                (*p_b)[i] -= lr * (*p_db)[i];
            }
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
        Tensor<T>* get_dW() {
            return this->p_dW;
        }
        int getWeightSize() {
            if (this->p_W != nullptr) return this->p_W->getSize();
            else return 0;
        }
        const int* getWeightShape() {
            if (this->p_W != nullptr) return this->p_W->getShape();
            else return nullptr;
        }
        Tensor<T>* get_db() {
            return this->p_db;
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
        Tensor<T>* p_col = nullptr;
    public:
        Conv2D(const int in_ch, const int out_ch,
               const int k=3, const int p=1, const int s=1, const char* cstr_ln="") : in_ch(in_ch), out_ch(out_ch), k(k), p(p), s(s), Layer<T>(cstr_ln) {
            int shape_buf[MAX_TENSOR_DIM] = { 0, };
            shape_buf[0] = out_ch; shape_buf[1] = in_ch; shape_buf[2] = k; shape_buf[3] = k;
            this->p_W  = new Tensor<T>(4, shape_buf);  // W[out_ch][in_ch][k][k]
            this->p_dW = new Tensor<T>(4, shape_buf);
            shape_buf[0] = out_ch;
            this->p_b  = new Tensor<T>(1, shape_buf);  // b[out_ch];
            this->p_db = new Tensor<T>(1, shape_buf);
        }
        ~Conv2D() {
            if (this->p_col) delete this->p_col;
        }
        virtual Tensor<T>* forward(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:Conv2D] Forward\n");
            #endif
            // Prepare
            this->p_x = p_x;
            Tensor<T>& x = *p_x;
            const int  x_dim   = x.getDim();      // Dimension: 4
            const int* x_shape = x.getShape();  // B, C, H, W
            const int B = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
            int out_h = 1 + (int)( (H - this->k + 2 * this->p) / this->s );
            int out_w = 1 + (int)( (W - this->k + 2 * this->p) / this->s );

            if (this->p_col) {
                delete p_col;
                this->p_col = nullptr;
            }
            this->p_col = this->im2col(p_x, this->k, this->s, this->p);    // col[B][in_ch][out_h][out_w][k][k]
            Tensor<T>& col = *this->p_col;

            // Eval
            this->y.dataInit();
            const int* y_shape = this->y.getShape();    // B, out_ch, out_h, out_w
            const int* col_shape = col.getShape();      // B, out_ch, out_h, out_w, k, k
            const int* W_shape = this->p_W->getShape(); // out_ch, in_ch, k, k
            const int* b_shape = this->p_b->getShape();
            int y_index[4] = { 0, };
            int col_index[6] = { 0, };
            int W_index[4] = { 0, };
            int b_index[1] = { 0, };
            for (int b = 0; b < B; b++) {
                y_index[0]   = b;
                col_index[0] = b;
                for (int o_ch = 0; o_ch < this->out_ch; o_ch++) {
                    y_index[1] = o_ch;
                    W_index[0] = o_ch;
                    b_index[0] = o_ch;
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
                            this->y[index_calc(4, y_shape, y_index)] += (*this->p_b)[index_calc(1, b_shape, b_index)];
                        }
                    }
                }
            }
            //if (this->y.shapeCheck())
            //delete p_col;
            return &(this->y);
        }
        virtual Tensor<T>* backward(Tensor<T>* p_dout) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:Conv2D] Backward\n");
            #endif
            this->p_dout = p_dout;
            Tensor<T>& dout = *p_dout;
            dout.reshape(this->y);
            Tensor<T>& db = *this->p_db;
            Tensor<T>& dW = *this->p_dW;
            Tensor<T>& col = *this->p_col;
            const int* dout_shape = dout.getShape();
            const int* db_shape = db.getShape();
            const int* dW_shape = dW.getShape();
            const int* col_shape = col.getShape();
            const int B = dout_shape[0], C = dout_shape[1], H = dout_shape[2], W = dout_shape[3];

            // Calculate derivative-bias (self.db = np.sum(dout, axis=0))
            int dout_index[4] = { 0, }; // dout[B][out_ch][out_h][out_w]
            int db_index[2] = { 0, };
            db.dataInit();
            for (int b = 0; b < B; b++) {
                dout_index[0] = b;
                db_index[0] = b;
                for (int c = 0; c < C; c++) {
                    dout_index[1] = c;
                    db_index[1] = c;
                    T sum = 0;
                    for (int h = 0; h < H; h++) {
                        dout_index[2] = h;
                        for (int w = 0; w < W; w++) {
                            dout_index[3] = w;
                            sum += dout[index_calc(4, dout_shape, dout_index)];
                        }
                    }
                    db[index_calc(2, db_shape, db_index)] = sum;
                }
            }

            // Calculate derivative-Weight (self.dW = np.dot(self.col.T, dout))
            int dW_index[4] = { 0, };   // dW[out_ch][in_ch][k][k]
            int col_index[6] = { 0, };  // col[B][in_ch][out_h][out_w][k][k] -> col.T[B][out_h][out_w][in_ch*k*k]
            dW.dataInit();
            for (int b = 0; b < B; b++) {
                dout_index[0] = b;
                col_index[0]  = b;
                for (int o_ch = 0; o_ch < C; o_ch++) {
                    dW_index[0] = o_ch;
                    dout_index[1] = o_ch;
                    for (int i_ch = 0; i_ch < in_ch; i_ch++) {
                        dW_index[1] = i_ch;
                        col_index[1] = i_ch;
                        for (int o_h = 0; o_h < H; o_h++) {
                            dout_index[2] = o_h;
                            col_index[2] = o_h;
                            for (int o_w = 0; o_w < W; o_w++) {
                                dout_index[3] = o_w;
                                col_index[3] = o_w;
                                for (int k_y = 0; k_y < this->k; k_y++) {
                                    dW_index[2] = k_y;
                                    col_index[4] = k_y;
                                    for (int k_x = 0; k_x < this->k; k_x++) {
                                        dW_index[3] = k_x;
                                        col_index[5] = k_x;
                                        dW[index_calc(4, dW_shape, dW_index)] += 
                                            dout[index_calc(4, dout_shape, dout_index)] * col[index_calc(6, col_shape, col_index)];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Calculate derivative-x(din) : dcol = np.dot(dout, self.col_W.T)
            Tensor<T>* p_dcol = new Tensor<T>(col.getDim(), col.getShape());
            Tensor<T>& dcol = *p_dcol;
            dcol.dataInit();
            for (int b = 0; b < B; b++) {
                col_index[0] = b;
                dout_index[0] = b;
                for (int o_ch = 0; o_ch < out_ch; o_ch++) {
                    dW_index[0] = o_ch;
                    dout_index[1] = o_ch;
                    for (int i_ch = 0; i_ch < in_ch; i_ch++) {
                        col_index[1] = i_ch;
                        dW_index[1] = i_ch;
                        for (int o_h = 0; o_h < H; o_h++) {
                            col_index[2] = o_h;
                            dout_index[2] = o_h;
                            for (int o_w = 0; o_w < W; o_w++) {
                                col_index[3] = o_w;
                                dout_index[3] = o_w;
                                for (int k_y = 0; k_y < this->k; k_y++) {
                                    col_index[4] = k_y;
                                    dW_index[2]  = k_y;
                                    for (int k_x = 0; k_x < this->k; k_x++) {
                                        col_index[5] = k_x;
                                        dW_index[3]  = k_x;
                                        dcol[index_calc(6, col_shape, col_index)] += dout[index_calc(4, dout_shape, dout_index)] * (*this->p_W)[index_calc(4, dW_shape, dW_index)];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // dcol = np.dot(dout, self.col_W.T)
            Tensor<T>* p_din = this->col2im(p_dcol, this->k, this->s, this->p);
            this->din.reshape(p_din->getDim(), p_din->getShape());
            memcpy(this->din.getData(), p_din->getData(), p_din->getSize()*sizeof(T));
            delete p_din;
            delete p_dcol;
            return &this->din;
        }
        virtual Tensor<T>* compile(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:LAYER:Conv2D] Compile\n");
            #endif
            this->p_x = p_x;
            const int x_dim = p_x->getDim();
            const int* x_shape = p_x->getShape();
            Tensor<T>& x = *p_x;
            const int B = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
            int out_h = 1 + ((H - this->k + 2 * this->p) / this->s);
            int out_w = 1 + ((W - this->k + 2 * this->p) / this->s);
            const int y_shape[] = { B, this->out_ch, out_h, out_w };
            this->y.reshape(4, y_shape);    // y[B][out_ch][out_h][out_w]
            this->din.reshape(x_dim, x_shape);
            return &this->y;
        }
    };

    template <typename T>
    class FullyConnected : Layer<T> {
    private:
        const int in_fmap, out_fmap;
    public:
        FullyConnected(const int in_fmap, const int out_fmap, const char* cstr_ln) : in_fmap(in_fmap), out_fmap(out_fmap), Layer<T>(cstr_ln) {
            int shape_buf[MAX_TENSOR_DIM] = { 0, };
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
                for (int o = 0; o < out_fmap; o++) {
                    this->y[b*out_fmap + o] += (*this->p_b)[o];
                }
            }
            return &this->y;
        }
        virtual Tensor<T>* backward(Tensor<T>* p_dout) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:FullyConnected] Backward\n");
            #endif
            this->p_dout = p_dout;
            Tensor<T>& dout = *p_dout;
            Tensor<T>& x = *this->p_x;
            Tensor<T>& W = *this->p_W;
            Tensor<T>& dW = *this->p_dW;
            Tensor<T>& db = *this->p_db;
            const int* din_shape = this->din.getShape();    // B, in_fmap
            const int* dout_shape = dout.getShape();        // B, out_fmap
            const int* x_shape = din_shape;
            const int* W_shape = W.getShape();              // in_fmap, out_fmap
            const int* db_shape = db.getShape();
            const int B = din_shape[0], O = din_shape[1];
            int din_index[2] = { 0, };
            int dout_index[2] = { 0, };
            int x_index[2] = { 0, };
            int W_index[2] = { 0, };

            // Weight Grad (dW = np.dot(x.T, dout))
            dW.dataInit();
            db.dataInit();
            for (int b = 0; b < B; b++) {
                dout_index[0] = b;
                x_index[0] = b;
                for (int i = 0; i < this->in_fmap; i++) {
                    W_index[0] = i;
                    x_index[1] = i;
                    for (int o = 0; o < this->out_fmap; o++) {
                        dout_index[1] = o;
                        W_index[1] = o;
                        dW[index_calc(2, W_shape, W_index)] += x[index_calc(2, x_shape, x_index)] * dout[index_calc(2, dout_shape, dout_index)];
                    }
                }
                for (int o = 0; o < this->out_fmap; o++) {
                    dout_index[1] = o;
                    db[o] += dout[index_calc(2, dout_shape, dout_index)];
                }
            }

            // Backprop
            this->din.dataInit();
            for (int b = 0; b < B; b++) {
                din_index[0] = b;
                dout_index[0] = b;
                for (int i = 0; i < this->in_fmap; i++) {
                    W_index[0] = i;
                    din_index[1] = i;
                    T tmp = 0;
                    for (int o = 0; o < this->out_fmap; o++) {
                        dout_index[1] = o;
                        W_index[1] = o;
                        tmp += dout[index_calc(2, dout_shape, dout_index)] * W[index_calc(2, W_shape, W_index)];
                    }
                    this->din[index_calc(2, din_shape, din_index)] = tmp;
                }
            }
            //Tensor<T>& din = *p_dout;
            return &this->din;
        }
        virtual Tensor<T>* compile(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:LAYER:FullyConnected] Compile\n");
            #endif
            const int x_dim = p_x->getDim();
            const int* x_shape = p_x->getShape();
            const int B = x_shape[0];
            int din_shape[2] = { B, 1 };
            for (int i = 1; i < x_dim; i++) din_shape[1] *= x_shape[i];
            const int y_shape[] = { x_shape[0], this->out_fmap };
            this->y.reshape(2, y_shape);
            this->din.reshape(2, din_shape);
            return &this->y;
        }
    };


    template <typename T>
    class ReLU : Layer<T> {
    private:
        Tensor<bool> mask;
    public:
        ReLU(const char* cstr_ln) : Layer<T>(cstr_ln) {

        }
        virtual Tensor<T>* forward(Tensor<T>* p_x) {
            this->p_x = p_x;
            Tensor<T>& x = *p_x;
            #ifdef LAYER_DEBUG
            printf("[DEBUG:ReLU] Forward\n");
            #endif
            const int x_size = x.getSize();
            for (int i = 0; i < x_size; i++) {
                if (x[i] < 0) {
                    this->y[i] = 0;
                    this->mask[i] = true;
                } else {
                    this->y[i] = x[i];
                    this->mask[i] = false;
                }
                //this->y[i] = x[i] < 0 ? 0 : x[i];
            }
            return &(this->y);
        }
        virtual Tensor<T>* backward(Tensor<T>* p_dout) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:ReLU] Backward\n");
            #endif
            this->p_dout = p_dout;
            Tensor<T>& dout = *p_dout;
            this->din = dout;
            const int dout_size = dout.getSize();
            for (int i = 0; i < dout_size; i++) {
                if (this->mask[i]) this->din[i] = 0;
            }
            return &this->din;
        }
        virtual Tensor<T>* compile(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:LAYER:ReLU] Compile\n");
            #endif
            const int x_dim = p_x->getDim();
            const int* x_shape = p_x->getShape();
            this->y.reshape(x_dim, x_shape);
            this->din.reshape(x_dim, x_shape);
            this->mask.reshape(x_dim, x_shape);
            return &(this->y);
        }
    };


    template <typename T>
    class MaxPool2D : Layer<T> {
    private:
        const int k, s; // k: kernel size, s: stride
        int out_h = 0, out_w = 0;
        Tensor<int> argmax; // Argument Max
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
            //out_h = ceilf((H - this->k) / this->s + 1);
            //out_w = ceilf((W - this->k) / this->s + 1);
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
                        out_index[2] = o_h;
                        for (int o_w = 0; o_w < out_w; o_w++) {
                            col_index[3] = o_w;
                            out_index[3] = o_w;
                            T max_val = -INFINITY;
                            int argmax = 0;
                            for (int k_y = 0; k_y < k; k_y++) {
                                col_index[4] = k_y;
                                for (int k_x = 0; k_x < k; k_x++) {
                                    col_index[5] = k_x;
                                    T tmp = col[index_calc(6, col_shape, col_index)];
                                    if (max_val < tmp) {
                                        max_val = tmp;
                                        argmax = k_y * k + k_x;
                                    }
                                }
                            }
                            this->y[index_calc(4, out_shape, out_index)] = max_val;
                            this->argmax[index_calc(4, out_shape, out_index)] = argmax;
                        }
                    }
                }
            }
            delete p_col;
            return &(this->y);
        }
        virtual Tensor<T>* backward(Tensor<T>* p_dout) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:MaxPool2D] Backward\n");
            #endif
            this->p_dout = p_dout;
            Tensor<T>& dout = *p_dout;
            const int* dout_shape = dout.getShape();    // B, C, out_h, out_w
            const int* din_shape = this->din.getShape();    // B, C, in_h, in_w
            const int B = dout_shape[0], C = dout_shape[1];
            int din_index[4] = { 0, };
            int dout_index[4] = { 0, };
            this->din.dataInit();
            for (int b = 0; b < B; b++) {
                din_index[0] = b;
                dout_index[0] = b;
                for (int c = 0; c < C; c++) {
                    din_index[1] = c;
                    dout_index[1] = c;
                    for (int h = 0; h < out_h; h++) {
                        dout_index[2] = h;
                        for (int w = 0; w < out_w; w++) {
                            dout_index[3] = w;
                            int argmax = this->argmax[index_calc(4, dout_shape, dout_index)];
                            int k_y = argmax / this->k, k_x = argmax % this->k;
                            din_index[2] = h * k + k_y, din_index[3] = w * k + k_x;
                            this->din[index_calc(4, din_shape, din_index)] = dout[index_calc(4, dout_shape, dout_index)];
                        }
                    }
                }
            }
            return &this->din;
        }
        virtual Tensor<T>* compile(Tensor<T>* p_x) {
            #ifdef LAYER_DEBUG
            printf("[DEBUG:LAYER:MaxPool2D] Compile\n");
            #endif
            const int x_dim = p_x->getDim();
            const int* x_shape = p_x->getShape();
            const int B = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
            out_h = ceilf((H - this->k) / this->s + 1);
            out_w = ceilf((W - this->k) / this->s + 1);
            const int y_shape[] = { B, C, out_h, out_w };
            this->y.reshape(4, y_shape);
            this->din.reshape(x_dim, x_shape);
            //this->p_dout->reshape(4, y_shape);
            this->p_dout = new Tensor<T>(4, y_shape);
            this->argmax.reshape(4, y_shape);
            return &this->y;
        }
    };

    template <typename T>
    class Criterion {
    private:
        string criterion_name;
    protected:
        Tensor<T>* p_pred = nullptr;
        Tensor<T>* p_t = nullptr;
        Tensor<T> din;
        float loss = 0;
    public:
        Criterion(const char* cstr_name) : criterion_name(cstr_name) {

        }
        ~Criterion() {
            if (this->p_pred) delete this->p_pred;
        }
        const char* getName() {
            return this->criterion_name.c_str();
        }
        virtual Tensor<T>* forward(Tensor<T>* p_x) {
            return p_x;
        }
        virtual float loss_func(Tensor<T>* p_t) {
            return 0;
        }
        virtual Tensor<T>* backward(Tensor<T>* p_dout) {
            return p_dout;
        }
        virtual Tensor<T>* compile(Tensor<T>* p_x) {
            return p_x;
        }
    };

    template <typename T>
    class SoftmaxWithLoss : Criterion<T> {
    private:
        Tensor<T>* Softmax(Tensor<T>* p_x) {
            Tensor<T>& x = *p_x;
            Tensor<T>& pred = *this->p_pred;
            const int x_dim = p_x->getDim();
            const int* x_shape = p_x->getShape();   // B, C
            const int B = x_shape[0], C = x_shape[1];
            int x_index[2] = { 0, };
            for (int b = 0; b < B; b++) {
                x_index[0] = b;
                // Get Max Value
                T maxVal = -INFINITY;
                for (int c = 0; c < C; c++) {
                    x_index[1] = c;
                    if (maxVal < x[index_calc(x_dim, x_shape, x_index)]) maxVal = x[index_calc(x_dim, x_shape, x_index)];
                }
                // Calculate Sum
                float sum = 0;
                for (int c = 0; c < C; c++) {
                    x_index[1] = c;
                    pred[index_calc(x_dim, x_shape, x_index)] = x[index_calc(x_dim, x_shape, x_index)] - maxVal;
                    pred[index_calc(x_dim, x_shape, x_index)] = expf(pred[index_calc(x_dim, x_shape, x_index)]);
                    sum = sum + pred[index_calc(x_dim, x_shape, x_index)];
                }
                // Calculate Prediction
                for (int c = 0; c < C; c++) {
                    x_index[1] = c;
                    pred[index_calc(x_dim, x_shape, x_index)] = pred[index_calc(x_dim, x_shape, x_index)] / sum;
                }
            }
            return this->p_pred;
        }
        float CrossEntropyError(Tensor<T>* p_t) {
            // p_t: Label Index
            this->p_t = p_t;
            const int* pred_shape = this->p_pred->getShape();
            const int B = pred_shape[0];
            int pred_index[2] = { 0, };
            for (int b = 0; b < B; b++) {
                int label = 0;
                pred_index[0] = b;
                for (int i = 0; i < 10; i++) {
                    if ((*p_t)[b * 10 + i] > 0.99) {
                        label = i;
                    }
                }
                pred_index[1] = label;  // Label Index
                this->loss -= logf((*this->p_pred)[index_calc(2, pred_shape, pred_index)]);
            }
            this->loss /= B;
            return this->loss;
        }

    public:
        SoftmaxWithLoss(const char* cstr_name) : Criterion<T>(cstr_name) {

        }
        ~SoftmaxWithLoss() {
            ~Criterion<T>();
        }        // Softmax & Cross Entropy Error
        virtual Tensor<T>* forward(Tensor<T>* p_x) {
            return this->Softmax(p_x);
        }
        virtual float loss_func(Tensor<T>* p_t) {
            return this->CrossEntropyError(p_t);
        }
        virtual Tensor<T>* backward(Tensor<T>* p_dout=nullptr) {
            Tensor<T>& pred = *this->p_pred;
            Tensor<T>& t = *this->p_t;
            const int* pred_shape = pred.getShape();
            const int* t_shape = this->p_t->getShape();
            const int B = t_shape[0], C = t_shape[1];
            int t_index[2] = { 0, };
            int pred_index[2] = { 0, };
            for (int b = 0; b < B; b++) {
                t_index[0] = b;
                pred_index[0] = b;
                for (int c = 0; c < C; c++) {
                    t_index[1] = c;
                    pred_index[1] = c;
                    this->din[index_calc(2, t_shape, t_index)] = (pred[index_calc(2, pred_shape, pred_index)] - t[index_calc(2, t_shape, t_index)]) / B;
                }
            }
            return &this->din;
        }
        // Compile
        virtual Tensor<T>* compile(Tensor<T>* p_x) {
            const int x_dim = p_x->getDim();
            const int* x_shape = p_x->getShape();   // B, C
            if (this->p_pred) delete this->p_pred;
            this->p_pred = new Tensor<float>(x_dim, x_shape);
            this->din.reshape(x_dim, x_shape);
            return this->p_pred;
        }
    };
}

#endif
 /** End of layer.hpp **/