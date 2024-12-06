package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strconv"
)

// PersistentData represents the Python class structure
type PersistentData struct {
	Dimensionality     int              `json:"dimensionality"`
	TotalElementsAdded int              `json:"total_elements_added"`
	MaxSeqID           int64            `json:"max_seq_id"`
	IDToLabel          map[string]int   `json:"id_to_label"`
	LabelToID          map[int]string   `json:"label_to_id"`
	IDToSeqID          map[string]int64 `json:"id_to_seq_id"`
}

func NewPersistentDataFromChromaMetadata(filename string) (*PersistentData, error) {
	// Create a Python script to read pickle and convert to JSON
	pythonScript := []byte(`
import pickle
import json
import sys
from typing import Dict, List, Optional, Sequence, Set, cast

SeqId = int

class PersistentData:
	"""Stores the data and metadata needed for a PersistentLocalHnswSegment"""
	
	dimensionality: Optional[int]
	total_elements_added: int
	
	max_seq_id: SeqId
	
	id_to_label: Dict[str, int]
	label_to_id: Dict[int, str]
	id_to_seq_id: Dict[str, SeqId]
	
	def __init__(
		self,
		dimensionality: Optional[int],
		total_elements_added: int,
		id_to_label: Dict[str, int],
		label_to_id: Dict[int, str],
		id_to_seq_id: Dict[str, SeqId],
	):
		self.dimensionality = dimensionality
		self.total_elements_added = total_elements_added
		self.id_to_label = id_to_label
		self.label_to_id = label_to_id
		self.id_to_seq_id = id_to_seq_id
	
	@staticmethod
	def load_from_file(filename: str) -> "PersistentData":
		"""Load persistent data from a file"""
		with open(filename, "rb") as f:
			ret = cast(PersistentData, pickle.load(f))
			return ret


def convert_pickle_to_json(pickle_file):
    pd = PersistentData.load_from_file(pickle_file)
    print(json.dumps({
		"dimensionality": pd.dimensionality,
		"total_elements_added": pd.total_elements_added,
		"id_to_label": pd.id_to_label,
		"label_to_id": pd.label_to_id,
		"id_to_seq_id": pd.id_to_seq_id
	}))

if __name__ == '__main__':
    convert_pickle_to_json(sys.argv[1])
`)

	// Write the Python script to a temporary file
	tmpFile, err := os.CreateTemp("", "pickle_converter_*.py")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.Write(pythonScript); err != nil {
		return nil, fmt.Errorf("failed to write Python script: %w", err)
	}
	tmpFile.Close()

	// Run the Python script
	cmd := exec.Command("python", tmpFile.Name(), filename)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("failed to run Python script: %w", err)
	}

	// Parse the JSON output
	var data PersistentData
	if err := json.Unmarshal(out.Bytes(), &data); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Convert string keys back to integers for LabelToID
	finalLabelToID := make(map[int]string)
	for k, v := range data.LabelToID {
		intKey, err := strconv.Atoi(strconv.Itoa(k))
		if err != nil {
			return nil, fmt.Errorf("failed to convert label key '%s': %w", k, err)
		}
		finalLabelToID[intKey] = v
	}

	// Create a new struct with the correct types
	return &PersistentData{
		Dimensionality:     data.Dimensionality,
		TotalElementsAdded: data.TotalElementsAdded,
		MaxSeqID:           data.MaxSeqID,
		IDToLabel:          data.IDToLabel,
		IDToSeqID:          data.IDToSeqID,
		LabelToID:          finalLabelToID, // Convert to the final type
	}, nil
}

func main() {
	data, err := NewPersistentDataFromChromaMetadata("/Users/tazarov/experiments/chroma/chroma-taz-22/2675/13126e01-622e-4c73-9fab-e10719a05011/index_metadata.pickle")
	if err != nil {
		fmt.Printf("Error loading pickle file: %v\n", err)
		return
	}

	// Print the loaded data
	fmt.Printf("Dimensionality: %v\n", data.Dimensionality)
	fmt.Printf("Total Elements Added: %d\n", data.TotalElementsAdded)
	fmt.Printf("Max Seq ID: %d\n", data.MaxSeqID)
	fmt.Printf("ID to Label: %v\n", data.IDToLabel)
	fmt.Printf("Label to ID: %v\n", data.LabelToID)
	fmt.Printf("ID to Seq ID: %v\n", data.IDToSeqID)
}
