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
	hnswIndex, err := NewIndex(WithPersistLocation(path))
	require.NoError(t, err, "Failed to create new index")
	t.Cleanup(func() {
		hnswIndex.Close()
	})
	_, err = os.Stat(path)
	require.NoError(t, err, "Index directory not created")
	_, err = os.Stat(filepath.Join(path, "config.json"))
	require.NoError(t, err, "Config file not created")
	idlist := []uint64{5, 10}
	err = hnswIndex.AddEmbeddings([][]float32{{1.0, 2.0, 3.3, 4.4, 5.0}, {5.1, 4.2, 3.3, 2.4, 1.5}}, idlist)
	require.NoError(t, err, "Failed to add embeddings")
	ids := hnswIndex.GetIDs()
	require.Equal(t, idlist, ids, "IDs mismatch")
	//err = hnswIndex.DeleteEmbeddings([]uint64{5})
	err = hnswIndex.DeleteEmbeddings(idlist)
	require.NoError(t, err, "Failed to delete embeddings")
	err = hnswIndex.Persist()
	require.NoError(t, err, "Failed to persist index")
	els := hnswIndex.GetActiveCount()
	fmt.Println(els)
}
