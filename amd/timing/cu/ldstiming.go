package cu

import "github.com/sarchlab/mgpusim/v5/amd/insts"

// LDSBankConflictCycles returns the additional execution cycles caused by
// LDS bank conflicts. GCN services the two 32-lane halves of a wave64
// independently, so the slower half determines the instruction latency.
func LDSBankConflictCycles(
	inst *insts.Inst,
	exec uint64,
	laneAddress func(lane int) uint32,
	bankCount, bankWidth, conflictPenalty int,
) int {
	if inst == nil || bankCount <= 0 || bankWidth <= 0 ||
		conflictPenalty <= 0 {
		return 0
	}

	maxConflict := 1
	for half := 0; half < 2; half++ {
		// A bank can broadcast one word to multiple lanes. Count distinct
		// bank-width words rather than lane requests.
		wordsByBank := make([]map[uint32]struct{}, bankCount)
		for lane := half * 32; lane < (half+1)*32; lane++ {
			if exec&(uint64(1)<<uint(lane)) == 0 {
				continue
			}

			for _, access := range ldsAccesses(inst, laneAddress(lane)) {
				firstWord := access.address / uint32(bankWidth)
				lastWord := (access.address + uint32(access.size) - 1) /
					uint32(bankWidth)
				for word := firstWord; word <= lastWord; word++ {
					bank := int(word % uint32(bankCount))
					if wordsByBank[bank] == nil {
						wordsByBank[bank] = make(map[uint32]struct{})
					}
					wordsByBank[bank][word] = struct{}{}
				}
			}
		}

		for _, words := range wordsByBank {
			if len(words) > maxConflict {
				maxConflict = len(words)
			}
		}
	}

	return (maxConflict - 1) * conflictPenalty
}

type ldsAccess struct {
	address uint32
	size    int
}

func ldsAccesses(inst *insts.Inst, base uint32) []ldsAccess {
	switch inst.Opcode {
	case 13, 54: // DS_{WRITE,READ}_B32
		return []ldsAccess{{base + inst.Offset0, 4}}
	case 14, 55: // DS_{WRITE,READ}2_B32
		return []ldsAccess{
			{base + inst.Offset0*4, 4},
			{base + inst.Offset1*4, 4},
		}
	case 15, 56: // DS_{WRITE,READ}2ST64_B32
		return []ldsAccess{
			{base + inst.Offset0*256, 4},
			{base + inst.Offset1*256, 4},
		}
	case 30, 57, 58: // DS_WRITE_B8, DS_READ_{I,U}8
		return []ldsAccess{{base + inst.Offset0, 1}}
	case 31, 59, 60: // DS_WRITE_B16, DS_READ_{I,U}16
		return []ldsAccess{{base + inst.Offset0, 2}}
	case 77, 118: // DS_{WRITE,READ}_B64
		return []ldsAccess{{base + inst.Offset0, 8}}
	case 78, 119: // DS_{WRITE,READ}2_B64
		return []ldsAccess{
			{base + inst.Offset0*8, 8},
			{base + inst.Offset1*8, 8},
		}
	case 79, 120: // DS_{WRITE,READ}2ST64_B64
		return []ldsAccess{
			{base + inst.Offset0*512, 8},
			{base + inst.Offset1*512, 8},
		}
	case 223, 255: // DS_{WRITE,READ}_B128
		return []ldsAccess{{base + inst.Offset0, 16}}
	default:
		// Unknown DS operations retain the configured base LDS latency.
		return nil
	}
}
