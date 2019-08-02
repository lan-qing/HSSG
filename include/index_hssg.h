#ifndef EFANNA2E_INDEX_HSSG_H
#define EFANNA2E_INDEX_HSSG_H

#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>

#include "index.h"
#include "neighbor.h"
#include "parameters.h"
#include "util.h"

namespace efanna2e {

    class IndexHSSG : public Index {
    public:
        explicit IndexHSSG(const size_t dimension, const size_t n, const size_t number_layer, Metric m,
                           Index *initializer);

        virtual ~IndexHSSG();

        virtual void Save(const char *filename) override;

        void Hier_save(const char *filename, const std::vector<unsigned> &num_layer, unsigned **down_link);

        virtual void Load(const char *filename) override;

        void Hier_load(const char *filename, std::vector<unsigned> &num_layer, unsigned **&down_link);

        virtual void Build(size_t n, const float *data,
                           const Parameters &parameters) override;

        virtual void Search(const float *query, const float *x, size_t k,
                            const Parameters &parameters, unsigned *indices) override;

        void SearchWithOptGraphPerLayer(const float *query, size_t K,
                                        const Parameters &parameters,
                                        unsigned *indices, std::vector<unsigned> starts,
                                        const std::vector<unsigned> &num_layer, unsigned layer);

        void SearchWithOptGraph(const float *query, size_t K,
                                const Parameters &parameters, unsigned *indices,
                                const std::vector<unsigned> &num_layer, unsigned ** down_link);

        void OptimizeGraph(float *data, const std::vector<unsigned> &num_layer, unsigned **down_link);

        void Hier_build(size_t n, float **data, unsigned **up_link, unsigned **down_link,
                        const std::vector<unsigned> &num_layer, const efanna2e::Parameters &parameters);

    protected:
        typedef std::vector<std::vector<unsigned>> CompactGraph;
        typedef std::vector<SimpleNeighbors> LockGraph;
        typedef std::vector<nhood> KNNGraph;

        CompactGraph *final_graph_;
        Index *initializer_;

        void init_graph(const Parameters &parameters, unsigned layer, unsigned n, float *h_data);

        void get_neighbors(const float *query, const Parameters &parameter, unsigned layer, unsigned n,
                           std::vector<Neighbor> &retset,
                           std::vector<Neighbor> &fullset, float *h_data);

        void get_neighbors(const unsigned q, const Parameters &parameter, unsigned layer, unsigned n,
                           std::vector<Neighbor> &pool, float *h_data);

        void sync_prune(unsigned q, std::vector<Neighbor> &pool,
                        const Parameters &parameters, float threshold,
                        SimpleNeighbor *cut_graph_, unsigned layer, unsigned num, float *h_data);

        void
        Link(const Parameters &parameters, SimpleNeighbor *cut_graph_, unsigned layer, unsigned num, float *h_data);

        void InterInsert(unsigned n, unsigned range, float threshold,
                         std::vector<std::mutex> &locks, SimpleNeighbor *cut_graph_, float *h_data);

        void Load_nn_graph(const char *filename);

        void strong_connect(const Parameters &parameter);

        void DFS(boost::dynamic_bitset<> &flag,
                 std::vector<std::pair<unsigned, unsigned>> &edges, unsigned root,
                 unsigned &cnt);

        bool check_edge(unsigned u, unsigned t);

        void findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                      const Parameters &parameter);

        void DFS_expand(const Parameters &parameter, const std::vector<unsigned> &num_layer, unsigned **down_link);

    private:
        unsigned width;
        unsigned ep_; //not in use
        unsigned n_layer;
        std::vector<unsigned> eps_;
        std::vector<std::mutex> locks;
        char *opt_graph_;
        char **hier_opt_graph_;
        size_t node_size;
        size_t data_len;
        size_t neighbor_len;
        KNNGraph nnd_graph;
    };

}  // namespace efanna2e

#endif  // EFANNA2E_INDEX_HSSG_H
