/**
 *
 * Deep Learning Seminar Task 2: Custom Neural Netweork Build with C++
 *
 * Written by Kwon, Han Kyul
 * File name: data_loader.hpp
 * Program date: 2022.02.18.
 *
 * Written on Feb 18, 2022
 * Modified on Feb 18, 2022
 * Modification History:
 *      1. Written by Hankyul Kwon
 *      2. Modified by Han Kyul Kwon on Feb 18, 2022
 *          (a) Add codes for normal execution.
 *
 * Compiler used: MSVC++ 14.16 (Visual Studio 2017 version 15.9)
 *
 */


#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stdio.h>
typedef char BYTE;

namespace data_loader {
    template <typename T>
    class DataLoader {
        private:
            FILE* fpbin = nullptr;
            T* data = nullptr;
            long size;  // Unit: N
            long fsize; // Unit: N * sizeof(T) Byte
        public:
        ~DataLoader() {
            if (data) delete data;
        }

        int copyDataOfRange(T* data, int from, int to) {
            if (size < to) {
                #ifdef DEBUG
                printf("[DEBUG:getDataOfRange] Get Data Out Of Range. (from:%d, to:%d)\n", from, to);
                #endif
                return 1;
            }
            memcpy(data, this->data + from, to - from);
            return 0;
        }

        int save_data(const char* cstr_fn) {

        }

        int load_data(const char* cstr_fn) {
            // Delete existing data
            if (data) delete data;

            // Open File
            if (fopen_s(&fpbin, cstr_fn, "rb")) {
                #ifdef DEBUG
                printf("[DEBUG:load_data] Failed to open %s.\n", cstr_fn);
                #endif
                return 1;
            }

            // Get File Size
            fseek(fpbin, 0L, SEEK_END);
            fsize = ftell(fpbin);
            fseek(fpbin, 0L, SEEK_SET);

            // Load Data
            size = fsize / sizeof(T);
            data = new T[size];
            fread(data, sizeof(BYTE), fsize, fpbin);
            fclose(fpbin);
            fpbin = nullptr;
            return 0;
        }
    };
}
#endif
 /** End of data_loader.hpp **/