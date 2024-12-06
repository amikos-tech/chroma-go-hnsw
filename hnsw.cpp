#include "hnswlib.h"
#include "vendors/hnswlib/hnswlib/hnswalg.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <string>

bool pathExists(const std::string &path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

HNSWIndex *create_index(HNSWIndexConfig *config) {
    std::unique_ptr <HNSWIndex> hnswIndex(new HNSWIndex());
    if (!hnswIndex) {
        throw std::runtime_error("Failed to allocate memory for index");
    }
    hnswIndex->config = config;
    // Create space on the heap
    std::unique_ptr <hnswlib::SpaceInterface<float>> space = nullptr;
    try {
        if (config->space == L2) {
            space.reset(new hnswlib::L2Space(config->dimension));
        } else if (config->space == IP) {
            space.reset(new hnswlib::InnerProductSpace(config->dimension));
        } else if (config->space == COSINE) {
            space.reset(new hnswlib::InnerProductSpace(config->dimension));
        } else {
            throw std::runtime_error("Invalid space type");
        }

        // safepoint to release ownership of the pointers
        hnswIndex->space = space.release();
        return hnswIndex.release();
    } catch (const std::exception &e) {
        fprintf(stderr, "Failed to create index: %s\n", e.what());
        throw;
    }
}

void init_index(HNSWIndex *index) {
    if (index == nullptr) {
        throw std::runtime_error("Index is null");
    }
    hnswlib::SpaceInterface<float> *space = (hnswlib::SpaceInterface<float> *) index->space;
    std::unique_ptr <hnswlib::HierarchicalNSW<float>> alg_hnsw = nullptr;
    std::string _persist_location(index->config->persist_location ? index->config->persist_location : "");
    std::size_t _max_elements(index->config->max_elements ? index->config->max_elements : 1000);
    std::size_t _M(index->config->M ? index->config->M : 16);
    std::size_t _ef_construction(index->config->ef_construction ? index->config->ef_construction : 200);
    std::size_t _random_seed(index->config->random_seed ? index->config->random_seed : 100);
    std::size_t _search_ef_default(index->config->search_ef_default ? index->config->search_ef_default : 10);
    auto _allow_replace_deleted(index->config->allow_replace_deleted ? index->config->allow_replace_deleted : false);
    auto _normalize(index->config->normalize ? index->config->normalize : false);
    auto _persist_on_write(index->config->persist_on_write ? index->config->persist_on_write : true);
    try {
        if (!_persist_location.empty()) {
            std::string header_path = _persist_location + "/header.bin"; // tight coupling?
            if (!pathExists(header_path)) {
                alg_hnsw.reset(new hnswlib::HierarchicalNSW<float>(
                        space,
                        _max_elements,
                        _M,
                        _ef_construction,
                        _random_seed,  // random seed
                        _allow_replace_deleted,  // allow replace deleted
                        _normalize,  // serialize mode
                        _persist_on_write,
                        _persist_location
                ));
            } else {
                alg_hnsw.reset(new hnswlib::HierarchicalNSW<float>(
                        space,
                        _persist_location,
                        false,
                        0,
                        _allow_replace_deleted,
                        _normalize,
                        _persist_on_write
                ));
                size_t dim = *((size_t *)alg_hnsw->dist_func_param_);
                index->config->M  = alg_hnsw->M_;
                index->config->ef_construction = alg_hnsw->ef_construction_;
                index->config->search_ef_default = alg_hnsw->ef_;
                index->config->max_elements = alg_hnsw->max_elements_;
                if (alg_hnsw->label_lookup_.size()>0){
                    auto it = alg_hnsw->label_lookup_.begin();
                    auto first_key = it->first;
                    auto vec = alg_hnsw->getDataByLabel<float>(first_key);
                    fprintf(stderr, "dims22 %zu\n", vec.size());

                    index->config->dimension = vec.size();
                }
                fprintf(stderr, "Loaded index txttt from %zu\n", alg_hnsw->size_data_per_element_/(sizeof(float)));
            }
        } else {
            alg_hnsw.reset(new hnswlib::HierarchicalNSW<float>(
                    space,
                    _max_elements,
                    _M,
                    _ef_construction,
                    _random_seed,  // random seed
                    _allow_replace_deleted,  // allow replace deleted
                    _normalize,  // serialize mode
                    false,
                    ""
            ));
        }

        alg_hnsw->setEf(_search_ef_default);
        index->index = alg_hnsw.release(); //transfer ownership
        index->initialized = true;
    } catch (const std::exception &e) {
        fprintf(stderr, "Failed to initialize index: %s\n", e.what());
        throw;
    }

}

void free_index(HNSWIndex *index) {
    if (index) {
        if (index->index) {
            hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HœierarchicalNSW<float> *) index->index;
            if (index->config->persist_location && index->config->persist_on_write) {
                alg_hnsw->saveIndex(index->config->persist_location);
            }
            alg_hnsw->closePersistentIndex();
            fprintf(stderr, "Freeing index232232\n");
            delete static_cast<hnswlib::HierarchicalNSW<float> *>(index->index);
        }
        if (index->space) {
            delete static_cast<hnswlib::SpaceInterface<float> *>(index->space);
        }
        free(index);
    }
}


void add_embeddings(HNSWIndex *hnswIndex, const Embeddings embeddings, bool replace_deleted) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;
    fprintf(stderr, "Adding embeddings to index %zu\n", *((size_t *)alg_hnsw->dist_func_param_));
    hnswlib::SpaceInterface<float> *space = (hnswlib::SpaceInterface<float> *) hnswIndex->space;
    for (int i = 0; i < embeddings.count; i++) {
        float *vec = embeddings.data + i * embeddings.dims;
        std::vector<float> normalized_vector(embeddings.dims);
        alg_hnsw->normalize_vector(vec, normalized_vector.data(), embeddings.dims);
        alg_hnsw->addPoint((void *) normalized_vector.data(), (size_t) embeddings.labels[i], replace_deleted);

    }
//    for (int i = 0; i < embeddings.count; i++) {
//        std::vector<float> vector = alg_hnsw->getDataByLabel<float>((size_t) embeddings.labels[i]);
//    }

}

/**
 * Get the list of active IDs in the index
 * @param hnswIndex
 * @param ids
 */
void get_ids_list(HNSWIndex *hnswIndex, label_type *ids) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    std::vector <hnswlib::labeltype> _ids;
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;
    for (auto kv: alg_hnsw->label_lookup_) {
        _ids.push_back(kv.first);
    }
    memcpy(ids, _ids.data(), _ids.size() * sizeof(label_type));
}

/*
 * Get the list of active IDs in the index
 * @param hnswIndex
 * @param ids
 */
void get_active_ids_list(HNSWIndex *hnswIndex, label_type *ids) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    std::vector <hnswlib::labeltype> _ids;
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;
    for (auto kv: alg_hnsw->label_lookup_) {
        if (alg_hnsw->isMarkedDeleted(kv.second)) {
            continue;
        }
        _ids.push_back(kv.first);
    }
    memcpy(ids, _ids.data(), _ids.size() * sizeof(label_type));
}

void get_deleted_ids_list(HNSWIndex *hnswIndex, label_type *ids) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    std::vector <hnswlib::labeltype> _ids;
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;
    for (auto kv: alg_hnsw->label_lookup_) {
        if (!alg_hnsw->isMarkedDeleted(kv.second)) {
            continue;
        }
        _ids.push_back(kv.first);
    }
    memcpy(ids, _ids.data(), _ids.size() * sizeof(label_type));
}

size_t get_current_count(HNSWIndex *hnswIndex) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;
    return alg_hnsw->getCurrentElementCount();
}

size_t get_max_elements(HNSWIndex *hnswIndex) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;
    return alg_hnsw->getMaxElements();
}

size_t get_deleted_count(HNSWIndex *hnswIndex) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;
    return alg_hnsw->getDeletedCount();
}

void resize_index(HNSWIndex *hnswIndex, size_t new_size) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;
    alg_hnsw->resizeIndex(new_size);
}

void delete_embeddings(HNSWIndex *hnswIndex, label_type *ids, size_t count) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;
    for (int i = 0; i < count; i++) {
        alg_hnsw->markDelete(ids[i]);
    }
}

void save_index(HNSWIndex *hnswIndex, const char *path_to_index) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;
    alg_hnsw->saveIndex(path_to_index);
}

void persist_dirty(HNSWIndex *hnswIndex) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;
    alg_hnsw->persistDirty();
}


KNNQueryRequest *create_knn_query_request(float *embeddings, size_t k, size_t num_threads, size_t count, size_t dims,
                                          FilterFunction filter_function) {
    KNNQueryRequest *request = new KNNQueryRequest();
    request->k = k;
    request->num_threads = num_threads;
    request->count = count;
    request->dims = dims;
    request->filter_function = filter_function;  // Caller can set this later if needed
    request->queryEmbeddings = embeddings;

    return request;
}

void free_knn_query_request(KNNQueryRequest *request) {
    if (request) {
//        if (request->queryEmbeddings) {
//            free(request->queryEmbeddings);
//        }
        free(request);
    }
}

KNNQueryResponse *create_knn_query_response(size_t k, size_t count) {
    KNNQueryResponse *response = new KNNQueryResponse();
    response->k = k;
    response->count = count;
    // Allocate space for the distances array (k results per query)
    response->distances = new Pair[k * count];
    return response;
}

void free_knn_query_response(KNNQueryResponse *response) {
    if (response) {
        if (response->distances) free(response->distances);
        free(response);
    }
}

class CustomFilterFunctor : public hnswlib::BaseFilterFunctor {
    std::function<bool(hnswlib::labeltype)> filter;

public:
    explicit CustomFilterFunctor(const std::function<bool(hnswlib::labeltype)> &f) {
        filter = f;
    }

    bool operator()(hnswlib::labeltype id) {
        return filter(id);
    }
};

void knn_query(HNSWIndex *hnswIndex, const KNNQueryRequest *query_request, const KNNQueryResponse *query_response) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    // Access the hnswlib index
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;

    for (size_t i = 0; i < query_request->count; i++) {
        const float *vec = query_request->queryEmbeddings + i * query_request->dims;
        CustomFilterFunctor idFilter(query_request->filter_function);
        CustomFilterFunctor *p_idFilter = query_request->filter_function ? &idFilter : nullptr;
        // Perform k-NN search
        std::priority_queue <std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(
                vec, query_request->k, p_idFilter);

        // Validate result size
        if (result.size() != query_request->k) {
            fprintf(stderr, "Error: Insufficient results. Consider adjusting ef or M parameters.\n");
            return;
        }

        // Fill query_response with results in descending order of proximity
        for (int j = query_request->k - 1; j >= 0; j--) {
            auto &result_tuple = result.top();
            query_response->distances[i * query_request->k + j].distance = result_tuple.first;
            query_response->distances[i * query_request->k + j].label = result_tuple.second;
            fprintf(stderr, "Distance: %f, Label: %zu\n", result_tuple.first, result_tuple.second);
            result.pop();
        }
    }
}

void get_items_by_ids(HNSWIndex *hnswIndex, label_type *ids, size_t ids_count, float *output_data, size_t *output_items,
                      size_t *output_dims) {
    if (!hnswIndex->initialized) {
        throw std::runtime_error("Index is not initialized");
    }
    hnswlib::HierarchicalNSW<float> *alg_hnsw = (hnswlib::HierarchicalNSW<float> *) hnswIndex->index;
    hnswlib::SpaceInterface<float> *space = (hnswlib::SpaceInterface<float> *) hnswIndex->space;

    *output_dims = *((size_t *) space->get_dist_func_param());
    *output_items = ids_count;
    for (size_t i = 0; i < ids_count; i++) {
        size_t id = ids[i];
        std::vector<float> vector = alg_hnsw->template getDataByLabel<float>(id);
        fprintf(stderr, "Getting vector data for ID: %zu\n", id);
        fprintf(stderr, "Vector data size: %zu\n", vector.size());
        if (!vector.empty() && vector.size() == *output_dims) {
            // Copy data from vector into the output buffer
            memcpy(output_data + i * (*output_dims), vector.data(), (*output_dims) * sizeof(float));
        } else {
            // Handle invalid ID or unexpected vector size
            memset(output_data + i * (*output_dims), 0, (*output_dims) * sizeof(float));
        }
    }
}


bool testFilter(KNNQueryRequest *request, label_type label) {
    if (request->filter_function != NULL) {

        const bool res = request->filter_function(label);
        fprintf(stderr, "Filter result: %d\n", res);
        return res;
    }
    return false;
}