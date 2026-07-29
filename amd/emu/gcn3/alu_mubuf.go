package gcn3

import (
	"log"

	"github.com/sarchlab/mgpusim/v5/amd/emu"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

// runMUBUF executes buffer_load/store_dword{,xN} for VGPR spill / scratch.
// Address: base(SRSRC[1:0]) + soffset + OFFSET + (offen? VGPR : 0).
// ponytail: only dword widths used by gfx90c matrixmult spill; no idxen/format.
func (u *ALU) runMUBUF(state emu.InstEmuState) {
	inst := state.Inst()
	switch inst.Opcode {
	case 20:
		u.runMUBUFLoadDWord(state, 4)
	case 21:
		u.runMUBUFLoadDWord(state, 8)
	case 22:
		u.runMUBUFLoadDWord(state, 12)
	case 23:
		u.runMUBUFLoadDWord(state, 16)
	case 28:
		u.runMUBUFStoreDWord(state, 4)
	case 29:
		u.runMUBUFStoreDWord(state, 8)
	case 30:
		u.runMUBUFStoreDWord(state, 12)
	case 31:
		u.runMUBUFStoreDWord(state, 16)
	default:
		log.Panicf("Opcode %d for MUBUF format is not implemented", inst.Opcode)
	}
}

// mubufRSRC reads the 4-SGPR buffer descriptor and returns base VA, stride,
// and whether ADD_TID_ENABLE is set (word3 bit 23).
func (u *ALU) mubufRSRC(state emu.InstEmuState) (base, stride uint64, addTID bool) {
	inst := state.Inst()
	idx := inst.Base.Register.RegIndex()
	w0 := uint32(state.ReadOperand(insts.NewSRegOperand(idx, idx, 1), 0))
	w1 := uint32(state.ReadOperand(insts.NewSRegOperand(idx+1, idx+1, 1), 0))
	w3 := uint32(state.ReadOperand(insts.NewSRegOperand(idx+3, idx+3, 1), 0))
	base = uint64(w0) | (uint64(w1&0xffff) << 32)
	stride = uint64((w1 >> 16) & 0x3fff)
	addTID = (w3 & (1 << 23)) != 0
	return
}

func (u *ALU) mubufSOffset(state emu.InstEmuState) uint64 {
	inst := state.Inst()
	if inst.Offset == nil {
		return 0
	}
	switch inst.Offset.OperandType {
	case insts.IntOperand, insts.LiteralConstant:
		return uint64(uint32(inst.Offset.IntValue))
	default:
		return state.ReadOperand(inst.Offset, 0) & 0xFFFFFFFF
	}
}

func (u *ALU) mubufAddr(
	state emu.InstEmuState, laneID int,
	base, stride, soff uint64, addTID bool,
) uint64 {
	inst := state.Inst()
	addr := base + soff + uint64(inst.Offset0)
	if addTID {
		addr += uint64(laneID) * stride
	}
	if inst.Offen {
		addr += state.ReadOperand(inst.Addr, laneID) & 0xFFFFFFFF
	}
	if inst.Idxen {
		log.Panic("MUBUF idxen not implemented")
	}
	return addr
}

func (u *ALU) runMUBUFLoadDWord(state emu.InstEmuState, nbytes uint64) {
	inst := state.Inst()
	pid := state.PID()
	exec := state.EXEC()
	base, stride, addTID := u.mubufRSRC(state)
	soff := u.mubufSOffset(state)
	for i := 0; i < 64; i++ {
		if exec&(1<<uint(i)) == 0 {
			continue
		}
		addr := u.mubufAddr(state, i, base, stride, soff, addTID)
		buf := u.storageAccessor.Read(pid, addr, nbytes)
		state.WriteOperandBytes(inst.Dst, i, buf)
	}
}

func (u *ALU) runMUBUFStoreDWord(state emu.InstEmuState, nbytes int) {
	inst := state.Inst()
	pid := state.PID()
	exec := state.EXEC()
	base, stride, addTID := u.mubufRSRC(state)
	soff := u.mubufSOffset(state)
	for i := 0; i < 64; i++ {
		if exec&(1<<uint(i)) == 0 {
			continue
		}
		addr := u.mubufAddr(state, i, base, stride, soff, addTID)
		data := state.ReadOperandBytes(inst.Data, i, nbytes)
		u.storageAccessor.Write(pid, addr, data)
	}
}
