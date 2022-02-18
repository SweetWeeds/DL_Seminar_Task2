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


#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <fstream>

template <typename T>
int load_data(T* data, const char* cstr_fn) {
    std::ifstream is(cstr_fn, std::ifstream::binary);
    int length = 0;
    if (is) {
        is.seekg(0, is.end);
        length = (int)is.tellg();
        is.seekg(0, is.beg);
        data = new T[length];
        assert(data);
        is.read((char*)data, length);
        is.close();
    }
    return length;
}

#endif
 /** End of data_loader.hpp **/