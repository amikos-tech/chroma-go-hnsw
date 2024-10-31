#include "hnswlib.h"
#include "vendors/hnswlib/hnswlib/hnswalg.h"



HNSWIndex* create_index(HNSWIndexConfig* config) {
    HNSWIndex* hnswIndex = (HNSWIndex*)malloc(sizeof(HNSWIndex));
    if (!hnswIndex) {
        throw std::runtime_error("Failed to allocate memory for index");
    }

    std::string _persist_location(config->persist_location ? config->persist_location : "");

    // Create space on the heap
    hnswlib::SpaceInterface<float>* space = nullptr;
    hnswlib::HierarchicalNSW<float>* alg_hnsw = nullptr;

    try {
        if (config->space == L2) {
            space = new hnswlib::L2Space(config->dimension);
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(
                space,
                config->max_elements,
                config->M,
                config->ef_construction,
                100,  // random seed
                false,  // allow replace deleted
                false,  // serialize mode
                config->persist_on_write,
                _persist_location
            );
        } else if (config->space == IP) {
            space = new hnswlib::InnerProductSpace(config->dimension);
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(
                space,
                config->max_elements,
                config->M,
                config->ef_construction
            );
        } else if (config->space == COSINE) {
            space = new hnswlib::InnerProductSpace(config->dimension);
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(
                space,
                config->max_elements,
                config->M,
                config->ef_construction
            );
        } else {
            free(hnswIndex);
            throw std::runtime_error("Invalid space type");
        }

        alg_hnsw->setEf(10);
        hnswIndex->index = alg_hnsw;
        hnswIndex->space = space;

        if (!_persist_location.empty()) {
            alg_hnsw->saveIndex(config->persist_location);
        }

        return hnswIndex;
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to create index: %s\n", e.what());
        delete space;
        delete alg_hnsw;
        free(hnswIndex);
        throw;
    }
}

void free_index(HNSWIndex* index) {
    if (index) {
        if (index->index) {
            delete static_cast<hnswlib::HierarchicalNSW<float>*>(index->index);
        }
        if (index->space) {
            delete static_cast<hnswlib::SpaceInterface<float>*>(index->space);
        }
        free(index);
    }
}


void add_embeddings(HNSWIndex* hnswIndex, const Embeddings embeddings, bool replace_deleted) {
    hnswlib::HierarchicalNSW<float>* alg_hnsw = (hnswlib::HierarchicalNSW<float>*)hnswIndex->index;
    for (int i = 0; i < embeddings.count; i++) {
        const float* vec = embeddings.data + i * embeddings.dims;
        alg_hnsw->addPoint(vec, embeddings.labels[i], replace_deleted);
    }
}

void get_ids_list(HNSWIndex* hnswIndex, label_type *ids){
    hnswlib::HierarchicalNSW<float>* alg_hnsw = (hnswlib::HierarchicalNSW<float>*)hnswIndex->index;
    for (auto kv : alg_hnsw->label_lookup_)
    {
        ids[kv.second] = kv.first;
    }
}

size_t get_current_count(HNSWIndex* hnswIndex){
    hnswlib::HierarchicalNSW<float>* alg_hnsw = (hnswlib::HierarchicalNSW<float>*)hnswIndex->index;
    return alg_hnsw->getCurrentElementCount();
}

size_t get_max_elements(HNSWIndex* hnswIndex){
    hnswlib::HierarchicalNSW<float>* alg_hnsw = (hnswlib::HierarchicalNSW<float>*)hnswIndex->index;
    return alg_hnsw->getMaxElements();
}

size_t get_deleted_count(HNSWIndex* hnswIndex){
    hnswlib::HierarchicalNSW<float>* alg_hnsw = (hnswlib::HierarchicalNSW<float>*)hnswIndex->index;
    return alg_hnsw->getDeletedCount();
}

void resize_index(HNSWIndex* hnswIndex, size_t new_size){
    hnswlib::HierarchicalNSW<float>* alg_hnsw = (hnswlib::HierarchicalNSW<float>*)hnswIndex->index;
    alg_hnsw->resizeIndex(new_size);
}

void delete_embeddings(HNSWIndex* hnswIndex, label_type *ids, size_t count){
    hnswlib::HierarchicalNSW<float>* alg_hnsw = (hnswlib::HierarchicalNSW<float>*)hnswIndex->index;
    for (int i = 0; i < count; i++) {
        alg_hnsw->markDelete(ids[i]);
    }
}

void save_index(HNSWIndex* hnswIndex, const char* path_to_index){
    hnswlib::HierarchicalNSW<float>* alg_hnsw = (hnswlib::HierarchicalNSW<float>*)hnswIndex->index;
    alg_hnsw->saveIndex(path_to_index);
}

void persist_dirty(HNSWIndex* hnswIndex){
    hnswlib::HierarchicalNSW<float>* alg_hnsw = (hnswlib::HierarchicalNSW<float>*)hnswIndex->index;
    alg_hnsw->persistDirty();
}