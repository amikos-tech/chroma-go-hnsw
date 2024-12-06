package hnsw

/*
#cgo CXXFLAGS: -I. -Ivendors/hnswlib/hnswlib/ -Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -march=native
#cgo darwin LDFLAGS: -stdlib=libc++ -framework Accelerate
#cgo linux LDFLAGS: -lstdc++ -fopenmp -ftree-vectorize
#cgo windows LDFLAGS: -lkernel32
#include <stdlib.h>
#include <stdio.h>
#include "hnswlib.h"
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"unsafe"
)

type Space uint

const (
	L2     Space = iota
	IP     Space = iota
	COSINE Space = iota
)

type OptionNew func(index *HNSWIndexConfig) error
type LoadIndexOption func(index *HNSWIndexConfig) error

// WithSpaceNew sets the space for a new index
func WithSpaceNew(space Space) OptionNew {
	return func(index *HNSWIndexConfig) error {
		index.Space = space
		return nil
	}
}

// WithDimensionNew sets the dimension for a new index
func WithDimensionNew(dimension int) OptionNew {
	return func(index *HNSWIndexConfig) error {
		index.Dimension = dimension
		return nil
	}
}

// WithMaxElementsNew sets the max elements for a new index
func WithMaxElementsNew(maxElements uint64) OptionNew {
	return func(index *HNSWIndexConfig) error {
		index.MaxElements = maxElements
		return nil
	}
}

// WithMaxElementsLoad sets the max elements for an existing index
func WithMaxElementsLoad(maxElements uint64) LoadIndexOption {
	return func(index *HNSWIndexConfig) error {
		index.MaxElements = maxElements
		return nil
	}
}

// WithMNew sets the M for a new index
func WithMNew(m uint64) OptionNew {
	return func(index *HNSWIndexConfig) error {
		index.M = m
		return nil
	}
}

// WithEFConstructionNew sets the efConstruction for a new index
func WithEFConstructionNew(efConstruction uint64) OptionNew {
	return func(index *HNSWIndexConfig) error {
		index.EFConstruction = efConstruction
		return nil
	}
}

// WithPersistLocationNew sets the persistLocation for a new index
func WithPersistLocationNew(persistLocation string) OptionNew {
	return func(index *HNSWIndexConfig) error {
		index.PersistLocation = persistLocation
		return nil
	}
}

// WithPersistLocationLoad sets the persistLocation for an existing index
func WithPersistLocationLoad(persistLocation string) LoadIndexOption {
	return func(index *HNSWIndexConfig) error {
		index.PersistLocation = persistLocation
		return nil
	}
}

// WithEFSearchNew sets the default ef for a new index. This is applied to all searches that do not specify an ef
func WithEFSearchNew(efSearch uint64) OptionNew {
	return func(index *HNSWIndexConfig) error {
		index.EfSearch = efSearch
		return nil
	}
}

// WithEFSearchLoad sets the default ef for an existing index. This is applied to all searches that do not specify an ef
func WithEFSearchLoad(efSearch uint64) LoadIndexOption {
	return func(index *HNSWIndexConfig) error {
		index.EfSearch = efSearch
		return nil
	}
}

// WithNumThreadsNew sets the number of threads for a new index
func WithNumThreadsNew(numThreads uint64) OptionNew {
	return func(index *HNSWIndexConfig) error {
		index.NumThreads = numThreads
		return nil
	}
}

// WithNumThreadsLoad sets the number of threads for an existing index
func WithNumThreadsLoad(numThreads uint64) LoadIndexOption {
	return func(index *HNSWIndexConfig) error {
		index.NumThreads = numThreads
		return nil
	}
}

// WithAllowReplaceDeletedNew sets the allowReplaceDeleted for a new index
func WithAllowReplaceDeletedNew(allowReplaceDeleted bool) OptionNew {
	return func(index *HNSWIndexConfig) error {
		index.AllowReplaceDeleted = allowReplaceDeleted
		return nil
	}
}

// WithReadOnlyLoad sets the readOnly for an existing index
func WithReadOnlyLoad(readOnly bool) LoadIndexOption {
	return func(index *HNSWIndexConfig) error {
		index.ReadOnly = readOnly
		return nil
	}
}

// WithResizeFactorNew sets the resize factor for a new index
func WithResizeFactorNew(resizeFactor float32) OptionNew {
	return func(index *HNSWIndexConfig) error {
		index.ResizeFactor = resizeFactor
		return nil
	}
}

// HNSWIndex is a Go struct that holds a reference to the C++ HNSWIndex
type HNSWIndex struct {
	config      *HNSWIndexConfig
	index       *C.HNSWIndex
	initialized bool
}

type HNSWIndexConfig struct {
	Space               Space   `json:"space"`
	Dimension           int     `json:"dimension"`
	Normalize           bool    `json:"normalize"`
	MaxElements         uint64  `json:"maxElements"`
	M                   uint64  `json:"m"`
	EFConstruction      uint64  `json:"efConstruction"`
	PersistLocation     string  `json:"-"`
	EfSearch            uint64  `json:"efSearch"`
	NumThreads          uint64  `json:"numThreads"`
	PersistOnWrite      bool    `json:"persistOnWrite"`
	AllowReplaceDeleted bool    `json:"allowReplaceDeleted"`
	ReadOnly            bool    `json:"readOnly"`
	ResizeFactor        float32 `json:"resizeFactor"`
}

// NewIndex creates a new HNSWIndex
func NewIndex(opts ...OptionNew) (*HNSWIndex, error) {
	internalConfig := &HNSWIndex{
		config: &HNSWIndexConfig{
			Space:               L2,
			Normalize:           false,
			M:                   16,
			EFConstruction:      100,
			EfSearch:            10,
			MaxElements:         1000,
			PersistOnWrite:      true,
			AllowReplaceDeleted: false,
			ReadOnly:            false,
			ResizeFactor:        1.2,
			NumThreads:          uint64(runtime.NumCPU()),
		},
	}

	for _, opt := range opts {
		err := opt(internalConfig.config)
		if err != nil {
			return nil, err
		}
	}
	if internalConfig.config.PersistLocation != "" {
		if _, err := os.Stat(internalConfig.config.PersistLocation); os.IsNotExist(err) {
			err := os.MkdirAll(internalConfig.config.PersistLocation, 0750)
			if err != nil {
				return nil, err
			}
		} else {
			return nil, fmt.Errorf("error: persist location already exists")
		}
		internalConfig.config.PersistOnWrite = true
	}

	config, err := getCConfig(internalConfig)
	if err != nil {
		return nil, err
	}
	internalConfig.index = C.create_index(config)
	defer func() {
		err := internalConfig.dumpConfig()
		if err != nil {
			fmt.Println(err)
		}
	}()
	return internalConfig, nil

}
func (h *HNSWIndex) dumpConfig() error {
	file, err := os.Create(filepath.Join(h.config.PersistLocation, "config.json"))
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ") // Optional: for pretty-printing
	if err := encoder.Encode(h.config); err != nil {
		return err
	}
	return nil
}

func LoadHNSWIndex(space Space, dimensions int, opts ...LoadIndexOption) (*HNSWIndex, error) {
	config := &HNSWIndexConfig{
		Space:               space,
		Normalize:           false,
		PersistOnWrite:      false,
		AllowReplaceDeleted: false,
		ReadOnly:            false,
		Dimension:           dimensions,
	}
	for _, opt := range opts {
		err := opt(config)
		if err != nil {
			return nil, err
		}
	}

	if _, err := os.Stat(config.PersistLocation); os.IsNotExist(err) {
		return nil, fmt.Errorf("error: persist location does not exist")
	}
	// read the config file

	index := &HNSWIndex{
		config: config,
	}
	cconfig, err := getCConfig(index)
	if err != nil {
		return nil, err
	}
	index.index = C.create_index(cconfig)
	defer func() {
		err := index.dumpConfig()
		if err != nil {
			fmt.Println(err)
		}
	}()
	if !index.initialized {
		C.init_index(index.index)
		index.initialized = true
	}
	newCfg, err := loadCConfig(index)
	if err != nil {
		return nil, err
	}
	index.config = newCfg
	return index, nil
}

func getCConfig(index *HNSWIndex) (*C.HNSWIndexConfig, error) {
	cSpace, err := getCSpace(index.config.Space)
	if err != nil {
		return nil, err
	}
	return &C.HNSWIndexConfig{
		space:                 cSpace,
		dimension:             C.size_t(index.config.Dimension),
		max_elements:          C.size_t(index.config.MaxElements),
		M:                     C.size_t(index.config.M),
		ef_construction:       C.size_t(index.config.EFConstruction),
		persist_location:      C.CString(index.config.PersistLocation),
		persist_on_write:      C.bool(index.config.PersistOnWrite),
		allow_replace_deleted: C.bool(index.config.AllowReplaceDeleted),
	}, nil
}

func loadCConfig(index *HNSWIndex) (*HNSWIndexConfig, error) {
	if index.index == nil {
		return nil, fmt.Errorf("index is nil")
	}
	return &HNSWIndexConfig{
		Space:               Space(index.index.config.space),
		Dimension:           int(index.index.config.dimension),
		MaxElements:         uint64(index.index.config.max_elements),
		M:                   uint64(index.index.config.M),
		EFConstruction:      uint64(index.index.config.ef_construction),
		PersistLocation:     C.GoString(index.index.config.persist_location),
		PersistOnWrite:      bool(index.index.config.persist_on_write),
		AllowReplaceDeleted: bool(index.index.config.allow_replace_deleted),
	}, nil
}

func getCSpace(space Space) (C.enum_Space, error) {
	switch space {
	case L2:
		return C.L2, nil
	case IP:
		return C.IP, nil
	case COSINE:
		return C.COSINE, nil
	default:
		return C.L2, fmt.Errorf("invalid space")
	}
}

func (h *HNSWIndex) AddEmbeddings(data [][]float32, ids []uint64) error {
	if !h.initialized {
		C.init_index(h.index)
		h.initialized = true
	}
	if h.config.ReadOnly {
		return fmt.Errorf("index is read only")
	}
	if len(data) == 0 {
		return fmt.Errorf("data is empty")
	}

	if len(data) != len(ids) {
		return fmt.Errorf("data and ids length mismatch")
	}

	if h.config.Dimension != 0 {
		for _, d := range data {
			if len(d) != h.config.Dimension {
				return fmt.Errorf("dimension mismatch, expected %d, got %d", h.config.Dimension, len(d))
			}
		}
	}

	if !hasUniqueIDs(ids) {
		return fmt.Errorf("ids are not unique")
	}

	if h.GetElementCount()+uint64(len(data)) > h.GetMaxElements() {
		err := h.Resize(uint64(float32(h.GetMaxElements()) * h.config.ResizeFactor))
		if err != nil {
			return err
		}
	}

	dims := len(data[0])
	count := len(data)
	labelsPtr := C.malloc(C.size_t(count) * C.size_t(unsafe.Sizeof(C.label_type(0))))
	if labelsPtr == nil {
		return fmt.Errorf("failed to allocate memory for labels")
	}
	defer C.free(labelsPtr)

	labels := unsafe.Slice((*C.label_type)(labelsPtr), count)
	for i, id := range ids {
		labels[i] = C.label_type(id)
	}

	dataSize := count * dims

	dataPtr := C.malloc(C.size_t(dataSize) * C.size_t(unsafe.Sizeof(C.float(0))))
	if dataPtr == nil {
		return fmt.Errorf("failed to allocate memory for data")
	}
	defer C.free(dataPtr)

	dataSlice := unsafe.Slice((*C.float)(dataPtr), dataSize)
	for i, vec := range data {
		offset := i * dims
		for j, val := range vec {
			dataSlice[offset+j] = C.float(val)
		}
	}

	embeddings := C.Embeddings{
		count:  C.size_t(count),
		dims:   C.size_t(dims),
		labels: (*C.label_type)(labelsPtr),
		data:   (*C.float)(dataPtr),
	}

	C.add_embeddings(h.index, embeddings, C.bool(h.config.AllowReplaceDeleted))
	h.config.Dimension = dims

	defer func() {
		err := h.dumpConfig()
		if err != nil {
			fmt.Println(err)
		}
	}()
	return nil
}

// Close frees the HNSWIndex
func (h *HNSWIndex) Close() {
	if !h.initialized {
		return
	}
	C.free_index(h.index)
}

func (h *HNSWIndex) GetIDs() []uint64 {
	if !h.initialized {
		C.init_index(h.index)
		h.initialized = true
	}
	// Get the number of elements
	numElements := uint64(C.get_current_count(h.index))
	if numElements == 0 {
		return []uint64{}
	}

	// Allocate memory for labels
	labels := (*C.label_type)(C.malloc(C.size_t(numElements) * C.size_t(unsafe.Sizeof(C.label_type(0)))))
	defer C.free(unsafe.Pointer(labels))

	C.get_ids_list(h.index, labels)

	ids := make([]uint64, numElements)
	cLabels := (*[1 << 30]C.label_type)(unsafe.Pointer(labels))[:numElements:numElements]
	for i, label := range cLabels {
		ids[i] = uint64(label)
	}
	return ids
}

func (h *HNSWIndex) GetActiveIDs() []uint64 {
	if !h.initialized {
		C.init_index(h.index)
		h.initialized = true
	}
	// Get the number of elements
	numElements := h.GetActiveCount()

	// Allocate memory for labels
	labels := (*C.label_type)(C.malloc(C.size_t(numElements) * C.size_t(unsafe.Sizeof(C.label_type(0)))))
	defer C.free(unsafe.Pointer(labels))

	C.get_active_ids_list(h.index, labels)

	ids := make([]uint64, numElements)
	cLabels := (*[1 << 30]C.label_type)(unsafe.Pointer(labels))[:numElements:numElements]
	for i, label := range cLabels {
		ids[i] = uint64(label)
	}
	return ids
}

func (h *HNSWIndex) GetElementCount() uint64 {
	if !h.initialized {
		C.init_index(h.index)
		h.initialized = true
	}
	return uint64(C.get_current_count(h.index))
}

func (h *HNSWIndex) GetActiveCount() uint64 {
	if !h.initialized {
		C.init_index(h.index)
		h.initialized = true
	}
	return uint64(C.get_current_count(h.index)) - uint64(C.get_deleted_count(h.index))
}

func (h *HNSWIndex) GetDeletedCount() uint64 {
	if !h.initialized {
		C.init_index(h.index)
		h.initialized = true
	}
	return uint64(C.get_deleted_count(h.index))
}

func (h *HNSWIndex) GetMaxElements() uint64 {
	if !h.initialized {
		C.init_index(h.index)
		h.initialized = true
	}
	return uint64(C.get_max_elements(h.index))
}

func (h *HNSWIndex) Resize(newSize uint64) error {
	if !h.initialized {
		C.init_index(h.index)
		h.initialized = true
	}
	if h.config.ReadOnly {
		return fmt.Errorf("index is read only")
	}
	if newSize < h.GetMaxElements() {
		return fmt.Errorf("new size is less than current max element count")
	}
	C.resize_index(h.index, C.size_t(newSize))
	return nil
}

func (h *HNSWIndex) DeleteEmbeddings(ids []uint64) error {
	if !h.initialized {
		C.init_index(h.index)
		h.initialized = true
	}
	if h.config.ReadOnly {
		return fmt.Errorf("index is read only")
	}
	if len(ids) == 0 {
		return fmt.Errorf("ids is empty")
	}
	if !hasUniqueIDs(ids) {
		return fmt.Errorf("ids are not unique")
	}

	labelsPtr := C.malloc(C.size_t(len(ids)) * C.size_t(unsafe.Sizeof(C.label_type(0))))
	if labelsPtr == nil {
		return fmt.Errorf("failed to allocate memory for labels")
	}
	defer C.free(labelsPtr)

	labels := unsafe.Slice((*C.label_type)(labelsPtr), len(ids))
	for i, id := range ids {
		labels[i] = C.label_type(id)
	}
	C.delete_embeddings(h.index, (*C.label_type)(labelsPtr), C.size_t(len(ids)))
	return nil
}

func (h *HNSWIndex) Persist() error {
	if !h.initialized {
		C.init_index(h.index)
		h.initialized = true
	}
	if h.config.ReadOnly {
		return fmt.Errorf("index is read only")
	}
	C.persist_dirty(h.index)
	return nil
}

type Embedding []float32

type HNSWIndexQuery struct {
	embeddings []*Embedding
	k          uint64
	filterIDs  []uint64
	ef         uint64
}

type HNSWIndexQueryResult struct {
}

type QueryOption func(query *HNSWIndexQuery) error

func WithFilterIDs(filterIDs []uint64) QueryOption {
	return func(query *HNSWIndexQuery) error {
		query.filterIDs = filterIDs
		return nil
	}
}

func WithEF(ef uint64) QueryOption {
	return func(query *HNSWIndexQuery) error {
		query.ef = ef
		return nil
	}
}

var currentFilter func(int) bool

//export goFilterWrapper
func goFilterWrapper(label C.label_type) C.bool {
	return C.bool(currentFilter(int(label)))
}

func createKNNQueryRequestGo(k, numThreads, count, dims int, filter func(label int) bool, embeddings []float32) *C.KNNQueryRequest {
	// Allocate memory for query embeddings
	cEmbeddings := (*C.float)(C.malloc(C.size_t(len(embeddings)) * C.size_t(unsafe.Sizeof(C.float(0)))))
	embeddingSlice := (*[1 << 30]C.float)(unsafe.Pointer(cEmbeddings))[:len(embeddings):len(embeddings)]
	if cEmbeddings == nil {
		fmt.Println("Failed to allocate memory for embeddings")
		return nil
	}
	// Copy the embeddings into the allocated C memory
	for i, val := range embeddings {
		embeddingSlice[i] = C.float(val)
	}
	currentFilter = filter

	// Create the KNNQueryRequest struct
	request := &C.KNNQueryRequest{
		k:               C.size_t(k),
		num_threads:     C.size_t(numThreads),
		count:           C.size_t(count),
		dims:            C.size_t(dims),
		filter_function: (C.FilterFunction)(C.goFilterWrapper),
		queryEmbeddings: cEmbeddings,
	}

	return request
}

func freeKNNQueryRequest(request *C.KNNQueryRequest) {
	fmt.Println("REQ", request.queryEmbeddings)
	if request.queryEmbeddings != nil {
		C.free(unsafe.Pointer(request.queryEmbeddings))
	}
	C.free(unsafe.Pointer(request))
}

func toRawSlice(data [][]float32) []float32 {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil
	}
	// Get the pointer to the first element of the first slice
	ptr := unsafe.Pointer(&data[0][0])

	// Calculate total length of the raw data
	totalLen := 0
	for _, row := range data {
		totalLen += len(row)
	}

	// Construct the raw slice header
	return *(*[]float32)(unsafe.Pointer(&reflect.SliceHeader{
		Data: uintptr(ptr),
		Len:  totalLen,
		Cap:  totalLen,
	}))
}

type Pair struct {
	Distance float32
	Label    int32
}

type KNNQueryResponse struct {
	K         uint
	Count     uint
	Distances []Pair
}

func (h *HNSWIndex) Query(queryEmbeddings [][]float32, k uint64, queryOpts ...QueryOption) (*HNSWIndexQueryResult, error) {
	if !h.initialized {
		C.init_index(h.index)
		h.initialized = true
	}
	query := &HNSWIndexQuery{
		k:  k,
		ef: h.config.EfSearch,
	}

	myFilter := func(label int) bool {
		return label > 0
	}
	currentFilter = myFilter

	rawSlice := toRawSlice(queryEmbeddings)

	embeddings := (*C.float)(unsafe.Pointer(&rawSlice[0]))

	var req *C.KNNQueryRequest = C.create_knn_query_request(embeddings, C.size_t(k), 1, C.size_t(len(queryEmbeddings)), C.size_t(len(queryEmbeddings[0])), (C.FilterFunction)(C.goFilterWrapper))
	defer C.free_knn_query_request(req)
	var resp *C.KNNQueryResponse = C.create_knn_query_response(C.size_t(1), C.size_t(1))
	defer C.free_knn_query_response(resp)
	C.knn_query(h.index, req, resp)
	fmt.Println(req, query, resp)
	fmt.Println(C.testFilter(req, 11))
	unpackedResponse := unpackKNNQueryResponse(resp)
	fmt.Printf("Response: %+v\n", unpackedResponse)
	runtime.KeepAlive(myFilter)
	_ = req
	return nil, nil

}

func unpackKNNQueryResponse(cResponse *C.KNNQueryResponse) KNNQueryResponse {
	k := uint(cResponse.k)
	count := uint(cResponse.count)

	// Create a Go slice from the C array
	cDistances := unsafe.Slice(cResponse.distances, k*count)
	distances := make([]Pair, len(cDistances))
	for i, cDist := range cDistances {
		var cdist C.Pair = cDist
		distances[i] = Pair{
			Distance: float32(cdist.distance), // Use actual field names here
			Label:    int32(cdist.label),      // Use actual field names here
		}
	}

	return KNNQueryResponse{
		K:         k,
		Count:     count,
		Distances: distances,
	}
}

func (r *KNNQueryResponse) String() string {
	return fmt.Sprintf("K: %d, Count: %d, Distances: %v", r.K, r.Count, r.Distances)
}

func (h *HNSWIndex) GetData(ids []uint64) ([][]float32, error) {
	if !h.initialized {
		C.init_index(h.index)
		h.initialized = true
	}
	idsCount := len(ids)
	if idsCount == 0 {
		return nil, nil
	}

	// Convert Go slice to C array
	cIds := (*C.label_type)(C.malloc(C.size_t(idsCount) * C.size_t(unsafe.Sizeof(C.label_type(0)))))
	defer C.free(unsafe.Pointer(cIds))
	for i, id := range ids {
		(*(*[1 << 30]C.label_type)(unsafe.Pointer(cIds)))[i] = C.label_type(id)
	}

	// Get vector dimension from the HNSW index
	var outputDims C.size_t
	var outputItems C.size_t

	// Pre-allocate output buffer
	maxDim := h.config.Dimension // Assume a method to get vector dimensions
	outputData := (*C.float)(C.malloc(C.size_t(idsCount) * C.size_t(maxDim) * C.size_t(unsafe.Sizeof(C.float(0)))))
	defer C.free(unsafe.Pointer(outputData))

	// Call the C function
	C.get_items_by_ids(
		h.index,
		cIds,
		C.size_t(idsCount),
		outputData,
		&outputItems,
		&outputDims,
	)

	// Convert C array to Go slice
	goData := make([][]float32, int(outputItems))
	cData := (*[1 << 30]C.float)(unsafe.Pointer(outputData))[: idsCount*int(outputDims) : idsCount*int(outputDims)]
	for i := 0; i < int(outputItems); i++ {
		goData[i] = make([]float32, int(outputDims))
		for j := 0; j < int(outputDims); j++ {
			goData[i][j] = float32(cData[i*int(outputDims)+j])
		}
	}

	fmt.Println(goData)

	return goData, nil
}
