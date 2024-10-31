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
	"unsafe"
)

type Space uint

const (
	L2     Space = iota
	IP     Space = iota
	COSINE Space = iota
)

type Option func(index *HNSWIndexConfig) error

// WithSpace sets the space for the index
func WithSpace(space Space) Option {
	return func(index *HNSWIndexConfig) error {
		index.Space = space
		return nil
	}
}

// WithDimension sets the dimension for the index
func WithDimension(dimension int) Option {
	return func(index *HNSWIndexConfig) error {
		index.Dimension = dimension
		return nil
	}
}

// WithMaxElements sets the max elements for the index
func WithMaxElements(maxElements int) Option {
	return func(index *HNSWIndexConfig) error {
		index.MaxElements = maxElements
		return nil
	}
}

// WithM sets the M for the index
func WithM(m int) Option {
	return func(index *HNSWIndexConfig) error {
		index.M = m
		return nil
	}
}

// WithEFConstruction sets the efConstruction for the index
func WithEFConstruction(efConstruction int) Option {
	return func(index *HNSWIndexConfig) error {
		index.EFConstruction = efConstruction
		return nil
	}
}

// WithPersistLocation sets the persistLocation for the index
func WithPersistLocation(persistLocation string) Option {
	return func(index *HNSWIndexConfig) error {
		index.PersistLocation = persistLocation
		return nil
	}
}

// WithEFSearch sets the default ef for the index. This is applied to all searches that do not specify an ef
func WithEFSearch(efSearch int) Option {
	return func(index *HNSWIndexConfig) error {
		index.EfSearch = efSearch
		return nil
	}
}

// WithNumThreads sets the number of threads for the index
func WithNumThreads(numThreads int) Option {
	return func(index *HNSWIndexConfig) error {
		index.NumThreads = numThreads
		return nil
	}
}

// WithAllowReplaceDeleted sets the allowReplaceDeleted for the index
func WithAllowReplaceDeleted(allowReplaceDeleted bool) Option {
	return func(index *HNSWIndexConfig) error {
		index.AllowReplaceDeleted = allowReplaceDeleted
		return nil
	}
}

// WithReadOnly sets the readOnly for the index
func WithReadOnly(readOnly bool) Option {
	return func(index *HNSWIndexConfig) error {
		index.ReadOnly = readOnly
		return nil
	}
}

func WithResizeFactor(resizeFactor float32) Option {
	return func(index *HNSWIndexConfig) error {
		index.ResizeFactor = resizeFactor
		return nil
	}
}

// HNSWIndex is a Go struct that holds a reference to the C++ HNSWIndex
type HNSWIndex struct {
	config *HNSWIndexConfig
	index  *C.HNSWIndex
}

type HNSWIndexConfig struct {
	Space               Space   `json:"space"`
	Dimension           int     `json:"dimension"`
	Normalize           bool    `json:"normalize"`
	MaxElements         int     `json:"maxElements"`
	M                   int     `json:"m"`
	EFConstruction      int     `json:"efConstruction"`
	PersistLocation     string  `json:"-"`
	EfSearch            int     `json:"efSearch"`
	NumThreads          int     `json:"numThreads"`
	PersistOnWrite      bool    `json:"persistOnWrite"`
	AllowReplaceDeleted bool    `json:"allowReplaceDeleted"`
	ReadOnly            bool    `json:"readOnly"`
	ResizeFactor        float32 `json:"resizeFactor"`
}

// NewIndex creates a new HNSWIndex
func NewIndex(opts ...Option) (*HNSWIndex, error) {

	//config := &C.HNSWIndexConfig{
	//	space:            C.L2,
	//	dimension:        C.int(dim),
	//	max_elements:     C.int(maxElements),
	//	M:                C.int(M),
	//	ef_construction:  C.int(efConstruction),
	//	persist_location: C.CString("./index"),
	//}

	internalConfig := &HNSWIndex{
		config: &HNSWIndexConfig{
			Space:               L2,
			Normalize:           false,
			M:                   16,
			EFConstruction:      100,
			EfSearch:            10,
			MaxElements:         1000,
			PersistOnWrite:      false,
			AllowReplaceDeleted: false,
			ReadOnly:            false,
			ResizeFactor:        1.2,
		}}

	for _, opt := range opts {
		err := opt(internalConfig.config)
		if err != nil {
			return nil, err
		}
	}
	if internalConfig.config.PersistLocation != "" {
		if _, err := os.Stat(internalConfig.config.PersistLocation); os.IsNotExist(err) {
			err := os.MkdirAll(internalConfig.config.PersistLocation, os.ModePerm)
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
func LoadHNSWIndex(persistLocation string) (*HNSWIndex, error) {
	if _, err := os.Stat(persistLocation); os.IsNotExist(err) {
		return nil, fmt.Errorf("error: persist location does not exist")
	}
	// read the config file

	configFile, err := os.Open(filepath.Join(persistLocation, "config.json"))
	if err != nil {
		return nil, err
	}
	defer configFile.Close()

	decoder := json.NewDecoder(configFile)
	var config HNSWIndexConfig
	if err := decoder.Decode(&config); err != nil {
		return nil, err
	}

	return nil, nil
}

func getCConfig(index *HNSWIndex) (*C.HNSWIndexConfig, error) {
	cSpace, err := getCSpace(index.config.Space)
	if err != nil {
		return nil, err
	}
	return &C.HNSWIndexConfig{
		space:                 cSpace,
		dimension:             C.int(index.config.Dimension),
		max_elements:          C.int(index.config.MaxElements),
		M:                     C.int(index.config.M),
		ef_construction:       C.int(index.config.EFConstruction),
		persist_location:      C.CString(index.config.PersistLocation),
		persist_on_write:      C.bool(index.config.PersistOnWrite),
		allow_replace_deleted: C.bool(index.config.AllowReplaceDeleted),
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
	C.free_index(h.index)
}

func (h *HNSWIndex) GetIDs() []uint64 {
	// Get the number of elements
	numElements := uint64(C.get_current_count(h.index))
	ids := make([]uint64, numElements)
	if numElements == 0 {
		return ids
	}
	labels := (*C.label_type)(C.malloc(C.size_t(numElements) * C.size_t(unsafe.Sizeof(C.label_type(0)))))
	defer C.free(unsafe.Pointer(labels))

	C.get_ids_list(h.index, labels)
	for i := uint64(0); i < numElements; i++ {
		// Get pointer to the i-th element
		ptr := unsafe.Pointer(uintptr(unsafe.Pointer(labels)) + uintptr(i)*unsafe.Sizeof(C.label_type(0)))
		// Convert to label_type and then to int
		ids[i] = uint64(*(*C.label_type)(ptr))
	}

	return ids
}

func (h *HNSWIndex) GetElementCount() uint64 {
	return uint64(C.get_current_count(h.index))
}

func (h *HNSWIndex) GetActiveCount() uint64 {
	return uint64(C.get_current_count(h.index)) - uint64(C.get_deleted_count(h.index))
}

func (h *HNSWIndex) GetDeletedCount() uint64 {
	return uint64(C.get_deleted_count(h.index))
}

func (h *HNSWIndex) GetMaxElements() uint64 {
	return uint64(C.get_max_elements(h.index))
}

func (h *HNSWIndex) Resize(newSize uint64) error {
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
	if h.config.ReadOnly {
		return fmt.Errorf("index is read only")
	}
	C.persist_dirty(h.index)
	return nil
}
