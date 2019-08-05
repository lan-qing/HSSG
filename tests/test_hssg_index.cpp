//
// Created by tianfeng on 2019/7/25.
//

#include "index_random.h"
#include "index_hssg.h"
#include "util.h"

char filename[] = "/home/tianfeng/projects/SSG/datasets/siftsmall/siftsmall_base.fvecs";
char save_filename[] = "/home/tianfeng/projects/SSG/graphs/siftsmall.hssg";

int main(int argc, char **argv) {
    if (argc < 12) {
        std::cout
                << "./run data_file L R Angle n_layer K_knn L_knn iter_knn S_knn R_knn save_graph_file [seed]"
                << std::endl;
        exit(-1);
    }
    if (argc == 13)
    {
        unsigned seed = (unsigned)atoi(argv[12]);
        srand(seed);
        std::cout << "Using Seed " << seed << std::endl;
    }

    unsigned points_num, dim;
    float **data_load = nullptr;
    unsigned **up_link = nullptr, **down_link = nullptr;

    unsigned L = (unsigned) atoi(argv[2]);
    unsigned R = (unsigned) atoi(argv[3]);
    float A = (float) atof(argv[4]);
    unsigned n_layer = (unsigned) atoi(argv[5]);; // 1 < n_layer < points_num

    unsigned K_knn = (unsigned) atoi(argv[6]); //200;
    unsigned L_knn = (unsigned) atoi(argv[7]); //200;
    unsigned iter_knn = (unsigned) atoi(argv[8]); //12;
    unsigned S_knn = (unsigned) atoi(argv[9]); //10;
    unsigned R_knn = (unsigned) atoi(argv[10]); //100;


    std::cout << "Data Path: " << argv[1] << std::endl;
    std::cout << "L = " << L << ", ";
    std::cout << "R = " << R << ", ";
    std::cout << "Angle = " << A << std::endl;
    std::cout << "n_layer = " << n_layer << std::endl;

    std::cout << "K_knn " << K_knn << std::endl;
    std::cout << "L_knn = " << L_knn << std::endl;
    std::cout << "iter_knn " << iter_knn << std::endl;
    std::cout << "S_knn " << S_knn << std::endl;
    std::cout << "R_knn = " << R_knn << std::endl;

    std::cerr << "Output SSG Path: " << argv[11] << std::endl;

    std::vector<unsigned> num_layer = efanna2e::hier_load_data(argv[1], points_num, dim, n_layer, data_load, up_link,
                                                               down_link);
    //data_load = efanna2e::data_align(data_load, points_num, dim);

    efanna2e::IndexRandom init_index(dim, points_num);
    efanna2e::IndexHSSG index(dim, points_num, n_layer, efanna2e::L2,
                              (efanna2e::Index *) (&init_index));

    efanna2e::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<float>("A", A);
    paras.Set<unsigned>("Layer_number", n_layer);

    paras.Set<unsigned>("K_knn", K_knn);
    paras.Set<unsigned>("L_knn", L_knn);
    paras.Set<unsigned>("iter_knn", iter_knn);
    paras.Set<unsigned>("S_knn", S_knn);
    paras.Set<unsigned>("R_knn", R_knn);

    auto s = std::chrono::high_resolution_clock::now();
    index.Hier_build(points_num, data_load, up_link, down_link, num_layer, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Build Time: " << diff.count() << "\n";
    index.Hier_save(argv[11], num_layer, down_link);
    return 0;
}
