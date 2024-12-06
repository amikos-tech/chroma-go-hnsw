package hnsw

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"os"
	"path/filepath"
	"testing"
)

func TestNewIndexWithPersistence(t *testing.T) {
	tempDir := t.TempDir()
	path := filepath.Join(tempDir, "index")
	fmt.Println(path)
	hnswIndex, err := NewIndex(WithPersistLocationNew(path), WithDimensionNew(5))
	require.NoError(t, err, "Failed to create new index")
	t.Cleanup(func() {
		hnswIndex.Close()
	})
	_, err = os.Stat(path)
	require.NoError(t, err, "Index directory not created")
	_, err = os.Stat(filepath.Join(path, "config.json"))
	require.NoError(t, err, "Config file not created")
	idlist := []uint64{5, 10, 11, 12}
	err = hnswIndex.AddEmbeddings([][]float32{
		{1.0, 2.0, 3.3, 4.4, 5.0},
		{5.1, 4.2, 3.3, 2.4, 1.5},
		{1.1, 2.2, 3.4, 4.5, 5.6},
		{1.2, 2.3, 3.5, 4.6, 5.7},
	}, idlist)
	require.NoError(t, err, "Failed to add embeddings")
	ids := hnswIndex.GetIDs()
	require.ElementsMatch(t, idlist, ids, "IDs mismatch")
	_, err = hnswIndex.Query([][]float32{{1.0, 2.0, 3.3, 4.4, 5.0}}, 2)
	require.NoError(t, err, "Failed to query")
	fmt.Println("IDS ", ids)
	_, err = hnswIndex.GetData(ids)
	require.NoError(t, err, "Failed to get data")
	//err = hnswIndex.DeleteEmbeddings([]uint64{5})
	err = hnswIndex.DeleteEmbeddings(idlist)
	require.NoError(t, err, "Failed to delete embeddings")
	err = hnswIndex.Persist()
	require.NoError(t, err, "Failed to persist index")
	els := hnswIndex.GetActiveCount()
	fmt.Println("Active count", els)

}

func TestOpen(t *testing.T) {
	//tempDir := t.TempDir()
	path := filepath.Join("./", "index1")
	hnswIndex, err := NewIndex(WithPersistLocationNew(path), WithDimensionNew(5))
	require.NoError(t, err, "Failed to create new index")
	_, err = os.Stat(path)
	require.NoError(t, err, "Index directory not created")
	_, err = os.Stat(filepath.Join(path, "config.json"))
	require.NoError(t, err, "Config file not created")
	idlist := []uint64{5, 10, 11, 12}
	err = hnswIndex.AddEmbeddings([][]float32{
		{1.0, 2.0, 3.3, 4.4, 5.0},
		{5.1, 4.2, 3.3, 2.4, 1.5},
		{1.1, 2.2, 3.4, 4.5, 5.6},
		{1.2, 2.3, 3.5, 4.6, 5.7},
	}, idlist)

	require.NoError(t, err, "Failed to add embeddings")
	err = hnswIndex.Persist()
	require.NoError(t, err, "Failed to persist index")
	hnswIndex.GetData(idlist)
	hnswIndex.Close()

	hnswIndex, err = LoadHNSWIndex(L2, 5, WithPersistLocationLoad(path))
	require.NoError(t, err, "Failed to open index")

	ids := hnswIndex.GetIDs()
	require.ElementsMatch(t, idlist, ids, "IDs mismatch")
	hnswIndex.GetData(ids)
	t.Log(hnswIndex.config.Dimension)

}

func TestLoad(t *testing.T) {
	//tempDir := t.TempDir()
	path := filepath.Join("./", "index2")
	hnswIndex, err := LoadHNSWIndex(L2, 5, WithPersistLocationLoad(path))
	require.NoError(t, err, "Failed to open index")

	ids := hnswIndex.GetIDs()
	t.Log(ids)
	t.Log(hnswIndex.GetElementCount())
	t.Log(hnswIndex.GetData(ids))
	//require.ElementsMatch(t, idlist, ids, "IDs mismatch")
	//hnswIndex.GetData(ids)
	t.Log(hnswIndex.config.Dimension)

}

func TestDeleteID(t *testing.T) {
	tempDir := t.TempDir()
	path := filepath.Join(tempDir, "index1")
	hnswIndex, err := NewIndex(WithPersistLocationNew(path), WithDimensionNew(5))
	require.NoError(t, err, "Failed to create new index")
	_, err = os.Stat(path)
	require.NoError(t, err, "Index directory not created")
	_, err = os.Stat(filepath.Join(path, "config.json"))
	require.NoError(t, err, "Config file not created")
	idlist := []uint64{5, 10, 11, 12}
	err = hnswIndex.AddEmbeddings([][]float32{
		{1.0, 2.0, 3.3, 4.4, 5.0},
		{5.1, 4.2, 3.3, 2.4, 1.5},
		{1.1, 2.2, 3.4, 4.5, 5.6},
		{1.2, 2.3, 3.5, 4.6, 5.7},
	}, idlist)

	require.NoError(t, err, "Failed to add embeddings")
	err = hnswIndex.Persist()
	require.NoError(t, err, "Failed to persist index")
	err = hnswIndex.DeleteEmbeddings([]uint64{5})
	require.NoError(t, err, "Failed to delete embeddings")
	allCount := hnswIndex.GetElementCount()
	activeCount := hnswIndex.GetActiveCount()
	activeIds := hnswIndex.GetActiveIDs()
	allIds := hnswIndex.GetIDs()
	require.Equal(t, uint64(3), activeCount, "Active count mismatch")
	require.Equal(t, uint64(4), allCount, "Active count mismatch")
	require.ElementsMatch(t, []uint64{10, 11, 12}, activeIds, "Active IDs mismatch")
	require.ElementsMatch(t, []uint64{5, 10, 11, 12}, allIds, "All IDs mismatch")
}
