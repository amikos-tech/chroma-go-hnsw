#ifndef WRAPPER_H
#define WRAPPER_H
#if defined(_WIN32) || defined(_WIN64)
#define EXPORT_GO_WRAPPER __declspec(dllexport)
#else
#define EXPORT_GO_WRAPPER __attribute__((visibility("default")))
#endif

#include <stddef.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
enum Space {
    L2 = 0,
    IP = 1,
    COSINE = 2
};

typedef struct {
    enum Space space;
    int dimension;
    int max_elements;
    int M;
    int ef_construction;
    bool allow_replace_deleted;
    bool normalize;
    bool persist_on_write;
    size_t random_seed;
    const char* persist_location;
} HNSWIndexConfig;

typedef size_t label_type;
typedef struct {
    size_t count;
    size_t dims;
    label_type *labels;
    float *data;
} Embeddings;

typedef struct HNSWIndex {
    void* index;
    void* space;
} HNSWIndex;




EXPORT_GO_WRAPPER HNSWIndex* create_index(HNSWIndexConfig * config);
EXPORT_GO_WRAPPER void free_index(HNSWIndex* hnswIndex);
EXPORT_GO_WRAPPER void add_embeddings(HNSWIndex* hnswIndex, const Embeddings embeddings, bool replace_deleted);
EXPORT_GO_WRAPPER void get_ids_list(HNSWIndex* hnswIndex, label_type *ids);
EXPORT_GO_WRAPPER size_t get_current_count(HNSWIndex* hnswIndex);
EXPORT_GO_WRAPPER size_t get_max_elements(HNSWIndex* hnswIndex);
EXPORT_GO_WRAPPER size_t get_deleted_count(HNSWIndex* hnswIndex);
EXPORT_GO_WRAPPER void resize_index(HNSWIndex* hnswIndex, size_t new_size);
EXPORT_GO_WRAPPER void delete_embeddings(HNSWIndex* hnswIndex, label_type *ids, size_t count);
EXPORT_GO_WRAPPER void save_index(HNSWIndex* hnswIndex, const char* path_to_index);
EXPORT_GO_WRAPPER void persist_dirty(HNSWIndex* hnswIndex);

//EXPORT_GO_WRAPPER void addPoint(const void *data_point, labeltype label, bool replace_deleted = false);
//EXPORT_GO_WRAPPER void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability);
//EXPORT_GO_WRAPPER void setEf(HNSWIndex* hnswIndex, int ef);
//EXPORT_GO_WRAPPER void persistDirty();
//EXPORT_GO_WRAPPER void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0);
//EXPORT_GO_WRAPPER void markDelete(labeltype label);
//EXPORT_GO_WRAPPER void unmarkDelete(labeltype label);
//EXPORT_GO_WRAPPER  std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(const void *query_data, size_t k, void *isIdAllowed = nullptr)


#ifdef __cplusplus
}
#endif

#endif // WRAPPER_H