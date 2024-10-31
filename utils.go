package hnsw

func hasUniqueIDs(ids []uint64) bool {
	seen := make(map[uint64]struct{})

	for _, id := range ids {
		if _, exists := seen[id]; exists {
			return false
		}
		seen[id] = struct{}{}
	}

	return true
}
