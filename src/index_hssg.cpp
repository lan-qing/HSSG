#include "index_hssg.h"

#include <omp.h>
#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <chrono>
#include <cmath>
#include <queue>

#include "exceptions.h"
#include "parameters.h"
#include "index_knn_graph.h"
#include "index_random.h"

constexpr double kPi = std::acos(-1);
namespace efanna2e {

    IndexHSSG::IndexHSSG(const size_t dimension, const size_t n, const size_t layer_number,
                         Metric m, // n: Points number
                         Index *initializer)
            : Index(dimension, n, m), n_layer(layer_number), initializer_{initializer} {}

    IndexHSSG::~IndexHSSG() {}

    void IndexHSSG::init_graph(const Parameters &parameters, unsigned layer, unsigned n, float *h_data) {
        float *center = new float[dimension_];
        for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
        for (unsigned i = 0; i < n; i++) {
            for (unsigned j = 0; j < dimension_; j++) {
                center[j] += h_data[i * dimension_ + j];
            }
        }
        for (unsigned j = 0; j < dimension_; j++) {
            center[j] /= n;
        }
        std::vector<Neighbor> tmp, pool;
        get_neighbors(center, parameters, layer, n, tmp, pool, h_data);
        ep_ = tmp[0].id;  // For Compatibility
    }

    void IndexHSSG::get_neighbors(const unsigned q, const Parameters &parameter, unsigned layer, unsigned n,
                                  std::vector<Neighbor> &pool, float *h_data) {
        boost::dynamic_bitset<> flags{n, 0};
        unsigned L = parameter.Get<unsigned>("L");
        flags[q] = true;
        for (unsigned i = 0; i < final_graph_[layer][q].size(); i++) {
            unsigned nid = final_graph_[layer][q][i];
            for (unsigned nn = 0; nn < final_graph_[layer][nid].size(); nn++) {
                unsigned nnid = final_graph_[layer][nid][nn];
                if (flags[nnid]) continue;
                flags[nnid] = true;
                float dist = distance_->compare(h_data + dimension_ * q,
                                                h_data + dimension_ * nnid, dimension_);
                pool.push_back(Neighbor(nnid, dist, true));
                if (pool.size() >= L) break;
            }
            if (pool.size() >= L) break;
        }
    }

    void IndexHSSG::get_neighbors(const float *query, const Parameters &parameter, unsigned layer, unsigned num,
                                  std::vector<Neighbor> &retset,
                                  std::vector<Neighbor> &fullset, float *h_data) {
        unsigned L = parameter.Get<unsigned>("L");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());
        std::mt19937 rng(rand());
        GenRandom(rng, init_ids.data(), L, (unsigned) num);
        boost::dynamic_bitset<> flags{num, 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= num) continue;
            // std::cout<<id<<std::endl;
            float dist = distance_->compare(h_data + dimension_ * (size_t) id, query,
                                            (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
            flags[id] = 1;
            L++;
        }
        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;
            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[layer][n].size(); ++m) {
                    unsigned id = final_graph_[layer][n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float dist = distance_->compare(query, h_data + dimension_ * (size_t) id,
                                                    (unsigned) dimension_);
                    Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance) continue;
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (L + 1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k) {
                k = nk;
            } else {
                ++k;
            }
        }
    }

    void IndexHSSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                               const Parameters &parameters, float threshold,
                               SimpleNeighbor *cut_graph_, unsigned layer, unsigned num, float *h_data) {
        unsigned range = parameters.Get<unsigned>("R");
        width = range;
        unsigned start = 0;

        boost::dynamic_bitset<> flags{num, 0};
        for (unsigned i = 0; i < pool.size(); ++i) {
            flags[pool[i].id] = 1;
        }
        for (unsigned nn = 0; nn < final_graph_[layer][q].size(); nn++) {
            unsigned id = final_graph_[layer][q][nn];
            if (flags[id]) continue;
            float dist = distance_->compare(h_data + dimension_ * (size_t) q,
                                            h_data + dimension_ * (size_t) id,
                                            (unsigned) dimension_);
            pool.push_back(Neighbor(id, dist, true));
        }

        std::sort(pool.begin(), pool.end());
        std::vector<Neighbor> result;
        if (pool[start].id == q) start++;
        result.push_back(pool[start]);

        while (result.size() < range && (++start) < pool.size()) {
            auto &p = pool[start];
            bool occlude = false;
            for (unsigned t = 0; t < result.size(); t++) {
                if (p.id == result[t].id) {
                    occlude = true;
                    break;
                }
                float djk = distance_->compare(h_data + dimension_ * (size_t) result[t].id,
                                               h_data + dimension_ * (size_t) p.id,
                                               (unsigned) dimension_);
                float cos_ij = (p.distance + result[t].distance - djk) / 2 /
                               sqrt(p.distance * result[t].distance);
                if (cos_ij > threshold) {
                    occlude = true;
                    break;
                }
            }
            if (!occlude) result.push_back(p);
        }

        SimpleNeighbor *des_pool = cut_graph_ + (size_t) q * (size_t) range;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < range) {
            des_pool[result.size()].distance = -1;
        }
    }

    void IndexHSSG::InterInsert(unsigned n, unsigned range, float threshold,
                                std::vector<std::mutex> &locks,
                                SimpleNeighbor *cut_graph_, float *h_data) {
        SimpleNeighbor *src_pool = cut_graph_ + (size_t) n * (size_t) range;
        for (size_t i = 0; i < range; i++) {
            if (src_pool[i].distance == -1) break;

            SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            SimpleNeighbor *des_pool = cut_graph_ + des * (size_t) range;

            std::vector<SimpleNeighbor> temp_pool;
            int dup = 0;
            {
                LockGuard guard(locks[des]);
                for (size_t j = 0; j < range; j++) {
                    if (des_pool[j].distance == -1) break;
                    if (n == des_pool[j].id) {
                        dup = 1;
                        break;
                    }
                    temp_pool.push_back(des_pool[j]);
                }
            }
            if (dup) continue;

            temp_pool.push_back(sn);
            if (temp_pool.size() > range) {
                std::vector<SimpleNeighbor> result;
                unsigned start = 0;
                std::sort(temp_pool.begin(), temp_pool.end());
                result.push_back(temp_pool[start]);
                while (result.size() < range && (++start) < temp_pool.size()) {
                    auto &p = temp_pool[start];
                    bool occlude = false;
                    for (unsigned t = 0; t < result.size(); t++) {
                        if (p.id == result[t].id) {
                            occlude = true;
                            break;
                        }
                        float djk = distance_->compare(
                                h_data + dimension_ * (size_t) result[t].id,
                                h_data + dimension_ * (size_t) p.id, (unsigned) dimension_);
                        float cos_ij = (p.distance + result[t].distance - djk) / 2 /
                                       sqrt(p.distance * result[t].distance);
                        if (cos_ij > threshold) {
                            occlude = true;
                            break;
                        }
                    }
                    if (!occlude) result.push_back(p);
                }
                {
                    LockGuard guard(locks[des]);
                    for (unsigned t = 0; t < result.size(); t++) {
                        des_pool[t] = result[t];
                    }
                    if (result.size() < range) {
                        des_pool[result.size()].distance = -1;
                    }
                }
            } else {
                LockGuard guard(locks[des]);
                for (unsigned t = 0; t < range; t++) {
                    if (des_pool[t].distance == -1) {
                        des_pool[t] = sn;
                        if (t + 1 < range) des_pool[t + 1].distance = -1;
                        break;
                    }
                }
            }
        }
    }

    void IndexHSSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_, unsigned layer, unsigned num,
                         float *h_data) {
        unsigned range = parameters.Get<unsigned>("R");
        std::vector<std::mutex> locks(num);

        float angle = parameters.Get<float>("A");
        float threshold = std::cos(angle / 180 * kPi);

#pragma omp parallel
        {
            // unsigned cnt = 0;
            std::vector<Neighbor> pool, tmp;
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < num; ++n) {
                pool.clear();
                tmp.clear();
                get_neighbors(n, parameters, layer, num, pool, h_data);
                sync_prune(n, pool, parameters, threshold, cut_graph_, layer, num, h_data);
                /*
                cnt++;
                if (cnt % step_size == 0) {
                  LockGuard g(progress_lock);
                  std::cout << progress++ << "/" << percent << " completed" << std::endl;
                }
                */
            }

#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < num; ++n) {
                InterInsert(n, range, threshold, locks, cut_graph_, h_data);
            }
        }
    }

    void IndexHSSG::Save(const char *filename) {

    }

    void IndexHSSG::Hier_save(const char *filename, const std::vector<unsigned> &num_layer, unsigned **down_link) {
        std::ofstream out(filename, std::ios::binary | std::ios::out);
        out.write((char *) &n_layer, sizeof(unsigned));
        out.write((char *) &width, sizeof(unsigned));
        for (unsigned j = 0; j < n_layer; ++j) {
            unsigned num_layer_tmp = num_layer[j];
            out.write((char *) &num_layer_tmp, sizeof(unsigned));

            if (num_layer_tmp != 1)
                assert(final_graph_[j].size() == num_layer_tmp);
            for (unsigned i = 0; i < num_layer[j]; ++i) {
                out.write((char *) &down_link[j][i], sizeof(unsigned));
            }

            for (unsigned i = 0; i < num_layer[j] && num_layer[j] > 1; i++) {
                unsigned GK = (unsigned) final_graph_[j][i].size();
                out.write((char *) &GK, sizeof(unsigned));
                out.write((char *) final_graph_[j][i].data(), GK * sizeof(unsigned));
            }
        }
        out.close();
    }

    void IndexHSSG::Load(const char *filename) {

    }

    void IndexHSSG::Hier_load(const char *filename, std::vector<unsigned> &num_layer, unsigned **&down_link) {
        std::ifstream in(filename, std::ios::binary);
        in.read((char *) &n_layer, sizeof(unsigned));
        in.read((char *) &width, sizeof(unsigned));
        num_layer.resize(n_layer);
        down_link = new unsigned *[n_layer];
        final_graph_ = new CompactGraph[n_layer];
        for (unsigned j = 0; j < n_layer; ++j) {
            unsigned num_layer_tmp;
            in.read((char *) &num_layer_tmp, sizeof(unsigned));
            num_layer[j] = num_layer_tmp;
            down_link[j] = new unsigned[num_layer[j]];
            // final_graph_[j].resize(num_layer_tmp);
            for (unsigned i = 0; i < num_layer[j]; ++i) {
                in.read((char *) &down_link[j][i], sizeof(unsigned));
            }

            for (unsigned i = 0; i < num_layer[j]; ++i) {
                unsigned k;
                in.read((char *) &k, sizeof(unsigned));
                if (in.eof()) break;
                std::vector<unsigned> tmp(k);
                in.read((char *) tmp.data(), k * sizeof(unsigned));
                final_graph_[j].push_back(tmp);
            }
        }
    }

    void IndexHSSG::Build(size_t n, const float *data, const Parameters &parameters)  // n: points_num
    {

    }

    void IndexHSSG::Hier_build(size_t n, float **data, unsigned **up_link, unsigned **down_link,
                               const std::vector<unsigned> &num_layer, const efanna2e::Parameters &parameters) {
        std::cout << "Totally " << n_layer << " layers. \n";
        unsigned range = parameters.Get<unsigned>("R");
        final_graph_ = new CompactGraph[n_layer];
        for (unsigned i = 0; i < n_layer - 1; ++i) {
            unsigned K_knn = parameters.Get<unsigned>("K_knn");
            if (num_layer[i] <= K_knn) {
                final_graph_[i].resize(num_layer[i]);
                for (unsigned j = 0; j < num_layer[i]; ++j) {
                    final_graph_[i][j].resize(num_layer[i] - 1);
                    for (unsigned k = 0; k < num_layer[i]; ++k) {
                        if (k == j) continue;
                        final_graph_[i][j].push_back(k);
                    }
                }
            } else {
                efanna2e::IndexRandom init_knn_index(dimension_, num_layer[i]);
                knn_efanna2e::IndexGraph knn_index(dimension_, num_layer[i], efanna2e::L2,
                                                   (efanna2e::Index *) (&init_knn_index));
                knn_index.Build(num_layer[i], data[i], parameters);
                final_graph_[i] = knn_index.ReturnFinalGraph();


                init_graph(parameters, i, num_layer[i], data[i]);
                SimpleNeighbor *cut_graph_ = new SimpleNeighbor[num_layer[i] * (size_t) range];
                Link(parameters, cut_graph_, i, num_layer[i], data[i]);
                final_graph_[i].resize(num_layer[i]);
                for (size_t k = 0; k < num_layer[i]; k++) {
                    SimpleNeighbor *pool = cut_graph_ + k * (size_t) range;
                    unsigned pool_size = 0;
                    for (unsigned j = 0; j < range; j++) {
                        if (pool[j].distance == -1) {
                            break;
                        }
                        pool_size = j;
                    }
                    ++pool_size;
                    final_graph_[i][k].resize(pool_size);
                    for (unsigned j = 0; j < pool_size; j++) {
                        final_graph_[i][k][j] = pool[j].id;
                    }
                }
            }
            std::cout << "Layer " << i << " built successfully!\n";
        }
        std::cout << "Start DFS...\n";
        DFS_expand(parameters, num_layer, down_link);
    }

    void
    IndexHSSG::DFS_expand(const Parameters &parameter, const std::vector<unsigned> &num_layer, unsigned **down_link) {
        unsigned range = parameter.Get<unsigned>("R"),
                K_knn = parameter.Get<unsigned>("K_knn");
        for (int k = n_layer - 1; k >= 0; --k) {
            if (K_knn >= num_layer[k])
                continue;
            boost::dynamic_bitset<> flags{num_layer[k], 0};
            std::queue<unsigned> myqueue;
            for (unsigned i = 0; i < num_layer[k + 1]; ++i) {
                myqueue.push(down_link[k + 1][i]);
                flags[down_link[k + 1][i]] = true;
            }
            std::vector<unsigned> uncheck_set(1);
            while (uncheck_set.size() > 0) {
                while (!myqueue.empty()) {
                    unsigned q_front = myqueue.front();
                    myqueue.pop();

                    for (unsigned j = 0; j < final_graph_[k][q_front].size(); j++) {
                        unsigned child = final_graph_[k][q_front][j];
                        if (flags[child])continue;
                        flags[child] = true;
                        myqueue.push(child);
                    }
                }

                uncheck_set.clear();
                for (unsigned j = 0; j < num_layer[k]; j++) {
                    if (flags[j])continue;
                    uncheck_set.push_back(j);
                }
                //std::cout <<i<<":"<< uncheck_set.size() << '\n';
                if (uncheck_set.size() > 0) {
                    for (unsigned j = 0; j < num_layer[k]; j++) {
                        if (flags[j] && final_graph_[k][j].size() < range) {
                            final_graph_[k][j].push_back(uncheck_set[0]);
                            break;
                        }
                    }
                    myqueue.push(uncheck_set[0]);
                    flags[uncheck_set[0]] = true;
                }
            }
        }
    }

    void IndexHSSG::OptimizeGraph(float *data, const std::vector<unsigned> &num_layer,
                                  unsigned **down_link) {  // use after build or load
        hier_opt_graph_ = new char *[n_layer];
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
        for (unsigned j = 0; j < n_layer; ++j) {
            data_ = data;
            data_len = (dimension_ + 1) * sizeof(float);
            neighbor_len = (width + 1) * sizeof(unsigned);
            node_size = data_len + neighbor_len;
            hier_opt_graph_[j] = (char *) malloc(node_size * num_layer[j]);
            for (unsigned i = 0; i < num_layer[j] - 1; i++) {
                char *cur_node_offset = hier_opt_graph_[j] + i * node_size;
                unsigned cur_data = i;
                for (int l = j - 1; l >= 0; --l) {
                    cur_data = down_link[l][cur_data];
                }
                const float *debug = data_ + cur_data * dimension_;
                float cur_norm = dist_fast->norm(debug, dimension_);

                std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
                std::memcpy(cur_node_offset + sizeof(float), data_ + cur_data * dimension_,
                            data_len - sizeof(float));

                cur_node_offset += data_len;
                unsigned k = final_graph_[j][i].size();
                std::memcpy(cur_node_offset, &k, sizeof(unsigned));
                std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[j][i].data(),
                            k * sizeof(unsigned));
                std::vector<unsigned>().swap(final_graph_[j][i]);
            }
            CompactGraph().swap(final_graph_[j]);
            data_ = nullptr;
        }
        free(data);
    }

    void IndexHSSG::Search(const float *query, const float *x, size_t K,
                           const Parameters &parameters, unsigned *indices) {

    }

    void IndexHSSG::SearchWithOptGraphPerLayer(const float *query, size_t K,
                                               const Parameters &parameters,
                                               unsigned *indices, std::vector<unsigned> starts,
                                               const std::vector<unsigned> &num_layer, unsigned layer) {
        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        std::mt19937 rng(rand());
        GenRandom(rng, init_ids.data(), L, (unsigned) num_layer[layer]);
        assert(starts.size() < L);
        for (unsigned i = 0; i < starts.size(); i++) {
            init_ids[i] = starts[i];
        }

        boost::dynamic_bitset<> flags{num_layer[layer], 0};
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= num_layer[layer]) continue;
            _mm_prefetch(hier_opt_graph_[layer] + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= num_layer[layer]) continue;
            float *x = (float *) (hier_opt_graph_[layer] + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout << L << std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(hier_opt_graph_[layer] + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (hier_opt_graph_[layer] + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(hier_opt_graph_[layer] + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float *data = (float *) (hier_opt_graph_[layer] + node_size * id);
                    float norm = *data;
                    data++;
                    float dist =
                            dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexHSSG::SearchWithOptGraph(const float *query, size_t K,
                                       const Parameters &parameters, unsigned *indices,
                                       const std::vector<unsigned> &num_layer, unsigned **down_link) {
        bool is_first_search_layer = true;
        std::vector<unsigned> tmp_results;
        for (int i = n_layer - 2; i >= 0; --i) {
            if (num_layer[i] < K) continue;
            if (is_first_search_layer) {
                tmp_results.resize(num_layer[i + 1]);
                for (unsigned j = 0; j < num_layer[i + 1]; ++j) {
                    tmp_results.push_back(down_link[i + 1][j]);
                }
                is_first_search_layer = false;
            } else {
                for (size_t k = 0; k < K; ++k) {
                    tmp_results[k] = down_link[i + 1][tmp_results[k]];
                }
            }
            std::vector<unsigned> tmp(K);
            SearchWithOptGraphPerLayer(query, K, parameters, tmp.data(), tmp_results, num_layer, i);
            tmp.swap(tmp_results);
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = tmp_results[i];
        }
    }
}
