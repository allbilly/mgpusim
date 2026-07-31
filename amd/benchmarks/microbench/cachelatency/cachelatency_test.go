package cachelatency

import "testing"

func TestBuildVectorChainsAreDisjointCycles(t *testing.T) {
	const (
		lanes          = 8
		nodesPerLane   = 7
		stride         = 16
		totalLineNodes = lanes * nodesPerLane
	)
	b := &Benchmark{
		ActiveLanes: lanes,
		Seed:        42,
		stride:      stride,
		m:           totalLineNodes,
		n:           totalLineNodes * stride,
	}

	b.buildVectorChains()

	if len(b.startIndices) != lanes {
		t.Fatalf("got %d starts, want %d", len(b.startIndices), lanes)
	}
	seen := make(map[uint32]bool)
	for lane, start := range b.startIndices {
		idx := start
		for step := 0; step < nodesPerLane; step++ {
			if idx%stride != 0 {
				t.Fatalf("lane %d index %d is not cache-line aligned", lane, idx)
			}
			if int(idx/stride)%lanes != lane {
				t.Fatalf("lane %d escaped to index %d", lane, idx)
			}
			if seen[idx] {
				t.Fatalf("index %d is shared or repeated before cycle end", idx)
			}
			seen[idx] = true
			idx = b.chain[idx]
		}
		if idx != start {
			t.Fatalf("lane %d did not close its cycle: got %d, want %d",
				lane, idx, start)
		}
	}
	if len(seen) != totalLineNodes {
		t.Fatalf("visited %d nodes, want %d", len(seen), totalLineNodes)
	}
}

func TestBuildVectorChainsDeterministic(t *testing.T) {
	makeBenchmark := func() *Benchmark {
		return &Benchmark{
			ActiveLanes: 4,
			Seed:        99,
			stride:      16,
			m:           32,
			n:           32 * 16,
		}
	}
	a, b := makeBenchmark(), makeBenchmark()
	a.buildVectorChains()
	b.buildVectorChains()

	for i := range a.chain {
		if a.chain[i] != b.chain[i] {
			t.Fatalf("chain differs at %d", i)
		}
	}
	for i := range a.startIndices {
		if a.startIndices[i] != b.startIndices[i] {
			t.Fatalf("start differs at lane %d", i)
		}
	}
}
