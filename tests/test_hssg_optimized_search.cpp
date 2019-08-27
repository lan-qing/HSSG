//
// Created by 付聪 on 2017/6/21.
//

#include <chrono>

#include "index_random.h"
#include "index_hssg.h"
#include "util.h"


void save_result(char *filename, std::vector<std::vector<unsigned> > &results) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);

    for (unsigned i = 0; i < results.size(); i++) {
        unsigned GK = (unsigned) results[i].size();
        out.write((char *) &GK, sizeof(unsigned));
        out.write((char *) results[i].data(), GK * sizeof(unsigned));
    }
    out.close();
}

int main(int argc, char **argv) {
    if (argc < 9) {
        std::cout << "./run data_file query_file hssg_path L K L_pre n_layer result_path [seed]"
                  << std::endl;
        exit(-1);
    }
    if (argc == 10) {
        unsigned seed = (unsigned)atoi(argv[9]);
        srand(seed);
        std::cerr << "Using Seed " << seed << std::endl;
    }

    unsigned L = (unsigned) atoi(argv[4]);
    unsigned K = (unsigned) atoi(argv[5]);
    unsigned L_pre = (unsigned) atoi(argv[6]);
    unsigned n_layer = (unsigned) atoi(argv[7]);

    std::cerr << "HSSG Path: " << argv[3] << std::endl;
    std::cerr << "Result Path: " << argv[8] << std::endl;
    std::cout << "L = " << L << ", ";
    std::cout << "K = " << K << ", ";
    std::cout << "L_pre = " << L_pre << ", ";
    std::cout << "n_layer = " << n_layer << "\n";

    unsigned points_num, dim;
    float *data_load = nullptr;
    data_load = efanna2e::load_data(argv[1], points_num, dim);
    data_load = efanna2e::data_align(data_load, points_num, dim);


    unsigned query_num, query_dim;
    float *query_load = nullptr;
    query_load = efanna2e::load_data(argv[2], query_num, query_dim);
    query_load = efanna2e::data_align(query_load, query_num, query_dim);

    assert(dim == query_dim);

    efanna2e::IndexRandom init_index(dim, points_num);
    efanna2e::IndexHSSG index(dim, points_num, n_layer, efanna2e::FAST_L2,
                              (efanna2e::Index *) (&init_index));

    // n_layer, num_layer, graph, down_link
    std::vector<unsigned> num_layer;
    auto **down_link = new unsigned *[n_layer];
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("K", K);
    paras.Set<unsigned>("L_pre", L_pre);
    index.Hier_load(argv[3], num_layer, down_link);
    index.OptimizeGraph(data_load, num_layer, down_link, paras);

    std::vector<std::vector<unsigned> > res(query_num);
    for (unsigned i = 0; i < query_num; i++) res[i].resize(K);
    std::chrono::duration<double> count(0);
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data(), num_layer, down_link, count);
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cerr << "Search Time: " << diff.count() << std::endl;
    std::cerr << "Count Time: " << count.count() << std::endl;
    save_result(argv[8], res);

    return 0;
}
