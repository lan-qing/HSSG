#include "util.h"

#include <malloc.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <cmath>

namespace efanna2e {

void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size, unsigned N) {
  for (unsigned i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  std::sort(addr, addr + size);
  for (unsigned i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  unsigned off = rng() % N;
  for (unsigned i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}

float* load_data(char* filename, unsigned& num, unsigned& dim) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Open file error" << std::endl;
    exit(-1);
  }

  in.read((char*)&dim, 4);

  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);

  float* data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * sizeof(float));
  }
  in.close();
  return data;
}

    std::vector<unsigned> hier_load_data(char *filename, unsigned &num, unsigned &dim, unsigned &layer_number, float **&data,
                           unsigned **&up_link, unsigned **&down_link) {

        ///////////////////////////////////////////////
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "Open file error" << std::endl;
            exit(-1);
        }

        in.read((char *) &dim, 4);

        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        size_t fsize = (size_t) ss;
        num = (unsigned) (fsize / (dim + 1) / 4);

        ////////////////////////////////////////////
        // layer[i]: which layer does point i belong to?
        // num_layer[i]: How many points does layer i have?
        std::vector<unsigned> layer(num), shuffle_list(num), num_layer(layer_number);
        for (unsigned i = 0; i < num; ++i) {
            shuffle_list[i] = i;
        }
        std::random_shuffle(shuffle_list.begin(), shuffle_list.end());
        num_layer[layer_number - 1] = 1;
        double quo = pow((float_t) num, 1 / (float_t) (layer_number - 1));
        for (unsigned i = layer_number - 2; i > 0; --i) {
            num_layer[i] = (unsigned) lround(num_layer[i + 1] * quo);
        }
        num_layer[0] = num;
        for (unsigned i = 0; i < layer_number; ++i) {
            for (unsigned j = 0; j < num_layer[i]; ++j) {
                layer[shuffle_list[j]] = i;
            }
        }
        ////////////////////////////////////////////////
        //float* data = new float[(size_t)num * (size_t)dim];
        data = new float *[layer_number];
        up_link = new unsigned *[layer_number];
        down_link = new unsigned *[layer_number];
        for (unsigned i = 0; i < layer_number; ++i) {
            data[i] = new float[num_layer[i] * dim];
            up_link[i] = new unsigned[num_layer[i]];
            down_link[i] = new unsigned[num_layer[i]];
        }

        in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num; ++i) {
            in.seekg(4, std::ios::cur);
            in.read((char *) (data[0] + i * dim), dim * sizeof(float));
        }
        in.close();
        auto *curs = new unsigned[layer_number];
        for (unsigned i = 0; i < layer_number; ++i) {
            curs[i] = 0;
            down_link[0][i] = i;
        }
        for (size_t i = 0; i < num; ++i) {
            curs[0] = i;
            for (unsigned j = 1; j <= layer[i]; ++j) {
                memcpy(data[j] + curs[j] * dim, data[0] + i * dim, dim * sizeof(float));
                down_link[j][curs[j]] = curs[j - 1];
                up_link[j - 1][curs[j - 1]] = curs[j];
                ++curs[j];
            }
        }
        delete []curs;
        for (unsigned i = 0; i < layer_number; ++i) {
            data[i] = data_align(data[i], num_layer[i], dim);
        }
        return num_layer;
    }


float* data_align(float* data_ori, unsigned point_num, unsigned& dim) {
#ifdef __GNUC__
#ifdef __AVX__
#define DATA_ALIGN_FACTOR 8
#else
#ifdef __SSE2__
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif
#endif
#endif

  float* data_new = 0;
  unsigned new_dim =
      (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
#ifdef __APPLE__
  data_new = new float[(size_t)new_dim * (size_t)point_num];
#else
  data_new =
      (float*)memalign(DATA_ALIGN_FACTOR * 4,
                       (size_t)point_num * (size_t)new_dim * sizeof(float));
#endif

  for (size_t i = 0; i < point_num; i++) {
    memcpy(data_new + i * new_dim, data_ori + i * dim, dim * sizeof(float));
    memset(data_new + i * new_dim + dim, 0, (new_dim - dim) * sizeof(float));
  }

  dim = new_dim;

#ifdef __APPLE__
  delete[] data_ori;
#else
  free(data_ori);
#endif

  return data_new;
}

}  // namespace efanna2e
