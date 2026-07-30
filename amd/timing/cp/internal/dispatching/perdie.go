package dispatching

import (
	"github.com/sarchlab/mgpusim/v5/amd/kernels"
	"github.com/sarchlab/mgpusim/v5/amd/protocol"
	"github.com/sarchlab/mgpusim/v5/amd/timing/cp/internal/resource"
)

// dieState tracks the dispatch progress of one XCD (die): the contiguous block
// of work-groups assigned to it and a round-robin cursor over its compute units.
type dieState struct {
	gridBuilder  kernels.GridBuilder // skipped to this die's WG block
	numWGInDie   int                 // size of this die's block
	dispatchedWG int                 // WGs already dispatched from the block
	currWG       *kernels.WorkGroup  // pulled from the block but not yet placed
	firstCU      int                 // index of this die's first CU in the pool
	numCUs       int                 // CUs assigned to this die (may differ per die)
	nextCUOffset int                 // round-robin CU cursor within the die
}

// perDieAlgorithm dispatches a kernel across numDies XCDs in parallel. The grid
// is split into numDies contiguous blocks (die-level block assignment); within a
// die the block's work-groups are spread round-robin (interleaved) across that
// die's compute units. The dispatcher drives each die independently through
// NextForDie and gates each with its own per-wavefront rate, so the dies run in
// parallel. CUs are assumed to be registered in XCD order, so die d owns the
// contiguous CU range [d*cusPerDie, (d+1)*cusPerDie).
//
// NOTE: the per-WG -> die policy is die-level block + intra-die interleave. The
// HW-accurate cross-XCD distribution is undocumented; this is a modeling choice
// and is irrelevant to uniform grids (e.g. empty_kernel). See issue #282.
type perDieAlgorithm struct {
	cuPool  resource.CUResourcePool
	numDies int

	numWG           int
	cusPerDie       int
	numDispatchedWG int
	dies            []*dieState
}

// RegisterCU allows the algorithm to dispatch work-groups to the CU.
func (a *perDieAlgorithm) RegisterCU(cu resource.DispatchableCU) {
	a.cuPool.RegisterCU(cu)
}

// NumDies returns the number of dies the algorithm dispatches across.
func (a *perDieAlgorithm) NumDies() int {
	return a.numDies
}

// StartNewKernel partitions the grid into one contiguous block per die.
// Work-groups are distributed proportionally to each die's CU count so a
// die with fewer CUs does not become the bottleneck.
func (a *perDieAlgorithm) StartNewKernel(info kernels.KernelLaunchInfo) {
	gb := kernels.NewGridBuilder()
	gb.SetKernel(info)
	a.numWG = gb.NumWG()
	a.numDispatchedWG = 0

	totalCUs := a.cuPool.NumCU()
	// Distribute CUs across dies with ceiling division so a remainder
	// (e.g. 7 CUs / 4 dies) does not silently waste CUs. Earlier dies get
	// the extra CU: 7/4 → dies get [2, 2, 2, 1].
	a.cusPerDie = (totalCUs + a.numDies - 1) / a.numDies

	// First pass: compute each die's CU count.
	cuCounts := make([]int, a.numDies)
	cuOffset := 0
	for d := 0; d < a.numDies; d++ {
		cusThisDie := a.cusPerDie
		if cuOffset+cusThisDie > totalCUs {
			cusThisDie = totalCUs - cuOffset
		}
		if cusThisDie < 1 {
			cusThisDie = 1
		}
		cuCounts[d] = cusThisDie
		cuOffset += cusThisDie
	}

	// Distribute WGs proportionally to CU count, using largest-remainder
	// allocation so the total is exact.
	wgCounts := make([]int, a.numDies)
	allocated := 0
	for d := 0; d < a.numDies; d++ {
		wgCounts[d] = a.numWG * cuCounts[d] / totalCUs
		allocated += wgCounts[d]
	}
	// Assign remaining WGs to dies with the largest fractional remainder.
	remainders := make([]int, a.numDies)
	for d := 0; d < a.numDies; d++ {
		remainders[d] = a.numWG*cuCounts[d] - wgCounts[d]*totalCUs
	}
	for allocated < a.numWG {
		best := 0
		for d := 1; d < a.numDies; d++ {
			if remainders[d] > remainders[best] {
				best = d
			}
		}
		wgCounts[best]++
		remainders[best] -= totalCUs
		allocated++
	}

	// Second pass: build die states.
	a.dies = make([]*dieState, a.numDies)
	cuOffset = 0
	wgStart := 0
	for d := 0; d < a.numDies; d++ {
		ds := &dieState{
			gridBuilder:  kernels.NewGridBuilder(),
			numWGInDie:   wgCounts[d],
			firstCU:      cuOffset,
			numCUs:       cuCounts[d],
			nextCUOffset: 0,
		}
		ds.gridBuilder.SetKernel(info)
		ds.gridBuilder.Skip(wgStart)
		a.dies[d] = ds
		cuOffset += cuCounts[d]
		wgStart += wgCounts[d]
	}
}

// NumWG returns the total number of work-groups across all dies.
func (a *perDieAlgorithm) NumWG() int {
	return a.numWG
}

// HasNext reports whether any die still has work-groups to dispatch.
func (a *perDieAlgorithm) HasNext() bool {
	return a.numDispatchedWG < a.numWG
}

// Next is unused on the per-die path (the dispatcher drives dies via
// NextForDie). It returns an invalid location.
func (a *perDieAlgorithm) Next() dispatchLocation {
	return dispatchLocation{}
}

// NextForDie returns the next work-group to dispatch on the given die, placed on
// the die's next compute unit (round-robin). It returns an invalid location when
// the die's block is exhausted or no CU on the die can currently host the WG (in
// which case the pulled WG is held and retried on a later call).
func (a *perDieAlgorithm) NextForDie(die int) dispatchLocation {
	ds := a.dies[die]
	if ds.dispatchedWG >= ds.numWGInDie {
		return dispatchLocation{}
	}

	if ds.currWG == nil {
		ds.currWG = ds.gridBuilder.NextWG()
	}

	for k := 0; k < ds.numCUs; k++ {
		cuOffset := (ds.nextCUOffset + k) % ds.numCUs
		cuID := ds.firstCU + cuOffset
		cu := a.cuPool.GetCU(cuID)

		locations, ok := cu.ReserveResourceForWG(ds.currWG)
		if !ok {
			continue
		}

		dispatch := dispatchLocation{
			valid: true,
			cu:    cu.DispatchingPort(),
			cuID:  cuID,
			wg:    ds.currWG,
		}
		dispatch.locations = make([]protocol.WfDispatchLocation, len(locations))
		for i, location := range locations {
			dispatch.locations[i] = protocol.WfDispatchLocation(location)
		}

		ds.currWG = nil
		ds.dispatchedWG++
		ds.nextCUOffset = cuOffset + 1
		a.numDispatchedWG++

		return dispatch
	}

	return dispatchLocation{}
}

// FreeResources marks the dispatched location's resources available.
func (a *perDieAlgorithm) FreeResources(location dispatchLocation) {
	a.cuPool.GetCU(location.cuID).FreeResourcesForWG(location.wg)
}
