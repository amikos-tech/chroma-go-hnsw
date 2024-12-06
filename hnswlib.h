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
    size_t dimension;
    size_t max_elements;
    size_t M;
    size_t search_ef_default;
    size_t ef_construction;
    bool allow_replace_deleted;
    bool normalize;
    bool persist_on_write;
    size_t random_seed;
    const char* persist_location;
} HNSWIndexConfig;

typedef size_t label_type;
typedef float dist_t;

typedef struct {
    size_t count;
    size_t dims;
    label_type *labels;
    float *data;
} Embeddings;

typedef struct HNSWIndex {
    void* index;
    void* space;
    HNSWIndexConfig *config;
    bool initialized;
} HNSWIndex;

typedef bool (*FilterFunction)(label_type label);
extern bool goFilterWrapper(label_type);

typedef struct {
    size_t k; // number of results to return
    size_t num_threads; // number of threads to use for the query
    size_t count; // number of query embeddings
    size_t dims; // dimensions of the query embeddings - this is the identical for each query embedding
    FilterFunction filter_function; // filter function that accepts a label and returns a boolean
    float *queryEmbeddings; // query embeddings
} KNNQueryRequest;

typedef struct {
    dist_t distance;
    label_type label;
} Pair;

typedef struct{
    size_t k;
    size_t count;
    Pair * distances;
} KNNQueryResponse;


EXPORT_GO_WRAPPER HNSWIndex* create_index(HNSWIndexConfig * config);
EXPORT_GO_WRAPPER void init_index(HNSWIndex *index);
EXPORT_GO_WRAPPER void free_index(HNSWIndex* hnswIndex);
EXPORT_GO_WRAPPER void add_embeddings(HNSWIndex* hnswIndex, const Embeddings embeddings, bool replace_deleted);
EXPORT_GO_WRAPPER void get_ids_list(HNSWIndex* hnswIndex, label_type *ids);
EXPORT_GO_WRAPPER void get_active_ids_list(HNSWIndex* hnswIndex, label_type *ids);
EXPORT_GO_WRAPPER void get_deleted_ids_list(HNSWIndex *hnswIndex, label_type *ids);
EXPORT_GO_WRAPPER size_t get_current_count(HNSWIndex* hnswIndex);
EXPORT_GO_WRAPPER size_t get_max_elements(HNSWIndex* hnswIndex);
EXPORT_GO_WRAPPER size_t get_deleted_count(HNSWIndex* hnswIndex);
EXPORT_GO_WRAPPER void resize_index(HNSWIndex* hnswIndex, size_t new_size);
EXPORT_GO_WRAPPER void delete_embeddings(HNSWIndex* hnswIndex, label_type *ids, size_t count);
EXPORT_GO_WRAPPER void save_index(HNSWIndex* hnswIndex, const char* path_to_index);
EXPORT_GO_WRAPPER void persist_dirty(HNSWIndex* hnswIndex);
EXPORT_GO_WRAPPER void get_items_by_ids(HNSWIndex* hnswIndex, label_type* ids, size_t ids_count, float* output_data, size_t* output_items, size_t *output_dims);

//EXPORT_GO_WRAPPER void set_ef();
//EXPORT_GO_WRAPPER void set_num_threads();

EXPORT_GO_WRAPPER void knn_query(HNSWIndex* hnswIndex, const KNNQueryRequest* query_request, const KNNQueryResponse* query_response);

EXPORT_GO_WRAPPER KNNQueryRequest* create_knn_query_request(float * embeddings,size_t k, size_t num_threads, size_t count, size_t dims, FilterFunction filter_function);
EXPORT_GO_WRAPPER void free_knn_query_request(KNNQueryRequest* request);

EXPORT_GO_WRAPPER KNNQueryResponse* create_knn_query_response(size_t k, size_t count);
EXPORT_GO_WRAPPER void free_knn_query_response(KNNQueryResponse* response);

//EXPORT_GO_WRAPPER void addPoint(const void *data_point, labeltype label, bool replace_deleted = false);
//EXPORT_GO_WRAPPER void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability);
//EXPORT_GO_WRAPPER void setEf(HNSWIndex* hnswIndex, int ef);
//EXPORT_GO_WRAPPER void persistDirty();
//EXPORT_GO_WRAPPER void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0);
//EXPORT_GO_WRAPPER void markDelete(labeltype label);
//EXPORT_GO_WRAPPER void unmarkDelete(labeltype label);
//EXPORT_GO_WRAPPER  std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(const void *query_data, size_t k, void *isIdAllowed = nullptr)

EXPORT_GO_WRAPPER bool testFilter(KNNQueryRequest* request, label_type label);

#ifdef __cplusplus
}
#endif

#endif // WRAPPER_H