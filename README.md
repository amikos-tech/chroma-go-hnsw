# About

A developer-friendly, Chroma HNSW lib wrapper.

## Why?

- Semantic router
- Semantic caching
- Index tooling
- Experimentation

## Design goals

- Minimalistic API
- Excellent DX
    - Index configuration should be persisted
    - Loading an index should be easy - just provide the path
    - ExternalID should be persistable as metadata - multiple metadata writers should be supported (defaults to Chroma's internal pickle based format)

## API

- ✅ Create index (in-memory or persisted)
- ✅ Add embeddings
- ✅ Delete embeddings
- ✅ Get Labels/IDs
- ✅ Persist index
- ✅ Get active embeddings count
- 🚫 Query embeddings
- 🚫 Index iterator


### Create new index

```go
hnswIndex, err := NewHNSWIndex(WithPersistLocation("./index")) //
	if err != nil {
		fmt.Println(err)
		return
	}
```

### Add Embeddings

```go
err = hnswIndex.AddEmbeddings([][]float32{{1.0, 2.0, 3.3, 4.4, 5.0}, {5.1, 4.2, 3.3, 2.4, 1.5}}, []uint64{5, 10})
```

