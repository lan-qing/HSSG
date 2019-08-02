//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_INDEX_GRAPH_H
#define EFANNA2E_INDEX_GRAPH_H

#include <cstddef>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include "util.h"
#include "parameters.h"
#include "knn_neighbor.h"
#include "index.h"


namespace knn_efanna2e {

    class IndexGraph : public efanna2e::Index {
    public:
        typedef std::vector<std::vector<unsigned> > CompactGraph;
        CompactGraph final_graph_;
        explicit IndexGraph(const size_t dimension, const size_t n, efanna2e::Metric m, Index *initializer);


        virtual ~IndexGraph();

        virtual void Save(const char *filename) override;

        virtual void Load(const char *filename) override;


        virtual void Build(size_t n, const float *data, const efanna2e::Parameters &parameters) override;

        virtual void Search(
                const float *query,
                const float *x,
                size_t k,
                const efanna2e::Parameters &parameters,
                unsigned *indices) override;

        void GraphAdd(const float *data, unsigned n, unsigned dim, const efanna2e::Parameters &parameters);

        void RefineGraph(const float *data, const efanna2e::Parameters &parameters);
        CompactGraph ReturnFinalGraph();

    protected:
        typedef std::vector<knn_efanna2e::nhood> KNNGraph;
        typedef std::vector<knn_efanna2e::LockNeighbor> LockGraph;

        Index *initializer_;
        KNNGraph graph_;



    private:
        void InitializeGraph(const efanna2e::Parameters &parameters);

        void InitializeGraph_Refine(const efanna2e::Parameters &parameters);

        void NNDescent(const efanna2e::Parameters &parameters);

        void join();

        void update(const efanna2e::Parameters &parameters);

        void generate_control_set(std::vector<unsigned> &c,
                                  std::vector<std::vector<unsigned> > &v,
                                  unsigned N);

        void eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set);

        void get_neighbor_to_add(const float *point, const efanna2e::Parameters &parameters, LockGraph &g,
                                 std::mt19937 &rng, std::vector<knn_efanna2e::Neighbor> &retset, unsigned n_total);

        void compact_to_Lockgraph(LockGraph &g);

        void parallel_graph_insert(unsigned id, knn_efanna2e::Neighbor nn, LockGraph &g, size_t K);

    };

}

#endif //EFANNA2E_INDEX_GRAPH_H
