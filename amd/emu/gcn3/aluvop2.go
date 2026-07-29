package gcn3

import (
	"log"
	"math"

	"github.com/sarchlab/mgpusim/v5/amd/bitops"
	"github.com/sarchlab/mgpusim/v5/amd/emu"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

// vop2WritesVCC reports whether a VOP2 opcode updates VCC (carry/borrow out).
// GFX9+ carry-out instructions (v_add_co_u32, v_addc_co_u32, etc.) use opcodes
// 25-30. GCN3 alternate encodings at opcodes 52-54 perform the same arithmetic
// but do not write VCC.
func vop2WritesVCC(opcode insts.Opcode) bool {
	switch opcode {
	case 25, 26, 27, 28, 29, 30:
		return true
	default:
		return false
	}
}

//nolint:gocyclo,funlen
func (u *ALU) runVOP2(state emu.InstEmuState) {
	inst := state.Inst()
	switch inst.Opcode {
	case 0:
		u.runVCNDMASKB32(state)
	case 1:
		u.runVADDF32(state)
	case 2:
		u.runVSUBF32(state)
	case 3:
		u.runVSUBREVF32(state)
	case 4:
		u.runVMULF32(state)
	case 5:
		u.runVMULF32(state)
	case 6:
		u.runVMULI32I24(state)
	case 8:
		u.runVMULU32U24(state)
	case 10:
		u.runVMINF32(state)
	case 11:
		u.runVMAXF32(state)
	case 12:
		u.runVMINI32(state)
	case 13:
		u.runVMAXI32(state)
	case 14:
		u.runVMINU32(state)
	case 15:
		u.runVMAXU32(state)
	case 16:
		u.runVLSHRREVB32(state)
	case 17:
		u.runVASHRREVI32(state)
	case 18:
		u.runVLSHLREVB32(state)
	case 19:
		u.runVANDB32(state)
	case 20:
		u.runVORB32(state)
	case 21:
		u.runVXORB32(state)
	case 22:
		u.runVMACF32(state)
	case 24:
		u.runVMADAKF32(state)
	case 25:
		// v_add_co_u32_e32 (GFX9+) / v_add_u32_e32 with carry out
		u.runVADDI32(state)
	case 26:
		// v_sub_co_u32_e32
		u.runVSUBI32(state)
	case 27:
		// v_subrev_co_u32_e32
		u.runVSUBREVI32(state)
	case 28:
		// v_addc_co_u32_e32
		u.runVADDCU32(state)
	case 29:
		u.runVSUBBU32(state)
	case 30:
		u.runVSUBBREVU32(state)
	case 38:
		// v_add_u16 (GFX9+)
		u.runVADDU16(state)
	case 42:
		u.runVLSHLREVB16(state)
	case 52:
		// v_add_u32_e32 (GCN3 encoding)
		u.runVADDI32(state)
	case 53:
		// v_sub_u32_e32 (GCN3 encoding)
		u.runVSUBI32(state)
	case 54:
		// v_subrev_u32_e32 (GCN3 encoding)
		u.runVSUBREVI32(state)
	default:
		log.Panicf("Opcode %d for VOP2 format (%s) is not implemented",
			inst.Opcode, insts.NewInstPrinter(nil).Print(inst))
	}
}

func (u *ALU) runVCNDMASKB32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		vcc := state.VCC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			if vcc&(1<<uint(i)) != 0 {
				state.WriteOperand(inst.Dst, i, state.ReadOperand(inst.Src1, i))
			} else {
				state.WriteOperand(inst.Dst, i, state.ReadOperand(inst.Src0, i))
			}
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVADDF32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			src0 := math.Float32frombits(uint32(state.ReadOperand(inst.Src0, i)))
			src1 := math.Float32frombits(uint32(state.ReadOperand(inst.Src1, i)))
			dst := src0 + src1
			state.WriteOperand(inst.Dst, i, uint64(math.Float32bits(dst)))
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode  %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVSUBF32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			src0 := math.Float32frombits(uint32(state.ReadOperand(inst.Src0, i)))
			src1 := math.Float32frombits(uint32(state.ReadOperand(inst.Src1, i)))
			dst := src0 - src1
			state.WriteOperand(inst.Dst, i, uint64(math.Float32bits(dst)))
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode  %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVSUBREVF32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			src0 := math.Float32frombits(uint32(state.ReadOperand(inst.Src0, i)))
			src1 := math.Float32frombits(uint32(state.ReadOperand(inst.Src1, i)))
			dst := src1 - src0
			state.WriteOperand(inst.Dst, i, uint64(math.Float32bits(dst)))
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode  %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVMULF32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			src0 := math.Float32frombits(uint32(state.ReadOperand(inst.Src0, i)))
			src1 := math.Float32frombits(uint32(state.ReadOperand(inst.Src1, i)))
			dst := src0 * src1
			state.WriteOperand(inst.Dst, i, uint64(math.Float32bits(dst)))
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVMULI32I24(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			src0 := int32(bitops.SignExt(
				bitops.ExtractBitsFromU64(state.ReadOperand(inst.Src0, i), 0, 23), 23))
			src1 := int32(bitops.SignExt(
				bitops.ExtractBitsFromU64(state.ReadOperand(inst.Src1, i), 0, 23), 23))

			dst := src0 * src1
			state.WriteOperand(inst.Dst, i, uint64(dst))
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVMULU32U24(state emu.InstEmuState) {
	inst := state.Inst()
	exec := state.EXEC()

	for i := 0; i < 64; i++ {
		if exec&(1<<uint(i)) == 0 {
			continue
		}

		src0 := (uint32(state.ReadOperand(inst.Src0, i)) << 8) >> 8
		src1 := (uint32(state.ReadOperand(inst.Src1, i)) << 8) >> 8
		dst := src0 * src1
		state.WriteOperand(inst.Dst, i, uint64(dst))
	}
}

func (u *ALU) runVMINF32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			src0 := math.Float32frombits(uint32(state.ReadOperand(inst.Src0, i)))
			src1 := math.Float32frombits(uint32(state.ReadOperand(inst.Src1, i)))
			dst := src0
			if src1 < src0 {
				dst = src1
			}

			state.WriteOperand(inst.Dst, i, uint64(math.Float32bits(dst)))
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVMAXF32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			src0 := math.Float32frombits(uint32(state.ReadOperand(inst.Src0, i)))
			src1 := math.Float32frombits(uint32(state.ReadOperand(inst.Src1, i)))
			dst := src0
			if src1 > src0 {
				dst = src1
			}

			state.WriteOperand(inst.Dst, i, uint64(math.Float32bits(dst)))
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVMINI32(state emu.InstEmuState) {
	inst := state.Inst()
	exec := state.EXEC()

	for i := 0; i < 64; i++ {
		if exec&(1<<uint(i)) == 0 {
			continue
		}

		var dst int32
		src0 := asInt32(uint32(state.ReadOperand(inst.Src0, i)))
		src1 := asInt32(uint32(state.ReadOperand(inst.Src1, i)))
		if src0 < src1 {
			dst = src0
		} else {
			dst = src1
		}

		state.WriteOperand(inst.Dst, i, uint64(dst))
	}
}

func (u *ALU) runVMAXI32(state emu.InstEmuState) {
	inst := state.Inst()
	exec := state.EXEC()

	for i := 0; i < 64; i++ {
		if exec&(1<<uint(i)) == 0 {
			continue
		}

		var dst int32
		src0 := asInt32(uint32(state.ReadOperand(inst.Src0, i)))
		src1 := asInt32(uint32(state.ReadOperand(inst.Src1, i)))
		if src0 > src1 {
			dst = src0
		} else {
			dst = src1
		}

		state.WriteOperand(inst.Dst, i, uint64(dst))
	}
}

func (u *ALU) runVMINU32(state emu.InstEmuState) {
	inst := state.Inst()
	exec := state.EXEC()

	for i := 0; i < 64; i++ {
		if exec&(1<<uint(i)) == 0 {
			continue
		}

		var dst uint32
		src0 := uint32(state.ReadOperand(inst.Src0, i))
		src1 := uint32(state.ReadOperand(inst.Src1, i))
		if src0 < src1 {
			dst = src0
		} else {
			dst = src1
		}

		state.WriteOperand(inst.Dst, i, uint64(dst))
	}
}

func (u *ALU) runVMAXU32(state emu.InstEmuState) {
	inst := state.Inst()
	exec := state.EXEC()

	for i := 0; i < 64; i++ {
		if exec&(1<<uint(i)) == 0 {
			continue
		}

		var dst uint32
		src0 := uint32(state.ReadOperand(inst.Src0, i))
		src1 := uint32(state.ReadOperand(inst.Src1, i))
		if src0 >= src1 {
			dst = src0
		} else {
			dst = src1
		}

		state.WriteOperand(inst.Dst, i, uint64(dst))
	}
}

func (u *ALU) runVLSHRREVB32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}
			src0 := state.ReadOperand(inst.Src0, i)
			src1 := state.ReadOperand(inst.Src1, i)
			dst := src1 >> (src0 & 0x1f)
			state.WriteOperand(inst.Dst, i, dst)
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVASHRREVI32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}
			src0 := uint32(state.ReadOperand(inst.Src0, i))
			src1 := int32(state.ReadOperand(inst.Src1, i))
			dst := src1 >> (src0 & 0x1f)
			state.WriteOperand(inst.Dst, i, uint64(dst))
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode  %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVLSHLREVB32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}
			src0 := uint32(state.ReadOperand(inst.Src0, i))
			src1 := uint32(state.ReadOperand(inst.Src1, i))
			dst := src1 << (src0 & 0x1f)
			state.WriteOperand(inst.Dst, i, uint64(dst))
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode  %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVANDB32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}
			src0 := uint32(state.ReadOperand(inst.Src0, i))
			src1 := uint32(state.ReadOperand(inst.Src1, i))
			dst := src0 & src1
			state.WriteOperand(inst.Dst, i, uint64(dst))
		}
	} else {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}
			src0 := u.sdwaSrcSelect(uint32(state.ReadOperand(inst.Src0, i)), inst.Src0Sel)
			src1 := u.sdwaSrcSelect(uint32(state.ReadOperand(inst.Src1, i)), inst.Src1Sel)
			dst := src0 & src1
			dst = u.sdwaDstSelect(uint32(state.ReadOperand(inst.Dst, i)), dst,
				inst.DstSel, inst.DstUnused)
			state.WriteOperand(inst.Dst, i, uint64(dst))
		}
	}
}

func (u *ALU) runVORB32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}
			src0 := uint32(state.ReadOperand(inst.Src0, i))
			src1 := uint32(state.ReadOperand(inst.Src1, i))
			dst := src0 | src1
			state.WriteOperand(inst.Dst, i, uint64(dst))
		}
	} else {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}
			src0 := u.sdwaSrcSelect(uint32(state.ReadOperand(inst.Src0, i)), inst.Src0Sel)
			src1 := u.sdwaSrcSelect(uint32(state.ReadOperand(inst.Src1, i)), inst.Src1Sel)
			dst := src0 | src1
			dst = u.sdwaDstSelect(uint32(state.ReadOperand(inst.Dst, i)), dst,
				inst.DstSel, inst.DstUnused)
			state.WriteOperand(inst.Dst, i, uint64(dst))
		}
	}
}

func (u *ALU) runVXORB32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}
			src0 := uint32(state.ReadOperand(inst.Src0, i))
			src1 := uint32(state.ReadOperand(inst.Src1, i))
			dst := src0 ^ src1
			state.WriteOperand(inst.Dst, i, uint64(dst))
		}
	} else {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}
			src0 := u.sdwaSrcSelect(uint32(state.ReadOperand(inst.Src0, i)), inst.Src0Sel)
			src1 := u.sdwaSrcSelect(uint32(state.ReadOperand(inst.Src1, i)), inst.Src1Sel)
			dst := src0 ^ src1
			dst = u.sdwaDstSelect(uint32(state.ReadOperand(inst.Dst, i)), dst,
				inst.DstSel, inst.DstUnused)
			state.WriteOperand(inst.Dst, i, uint64(dst))
		}
	}
}

func (u *ALU) runVMACF32(state emu.InstEmuState) {
	inst := state.Inst()

	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			dst := asFloat32(uint32(state.ReadOperand(inst.Dst, i)))
			src0 := asFloat32(uint32(state.ReadOperand(inst.Src0, i)))
			src1 := asFloat32(uint32(state.ReadOperand(inst.Src1, i)))
			dst += src0 * src1
			state.WriteOperand(inst.Dst, i, uint64(float32ToBits(dst)))
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode  %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVMADAKF32(state emu.InstEmuState) {
	inst := state.Inst()

	if !inst.IsSdwa {
		exec := state.EXEC()
		k := asFloat32(uint32(state.ReadOperand(inst.Src2, 0)))
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}
			src0 := asFloat32(uint32(state.ReadOperand(inst.Src0, i)))
			src1 := asFloat32(uint32(state.ReadOperand(inst.Src1, i)))
			dst := src0*src1 + k
			state.WriteOperand(inst.Dst, i, uint64(float32ToBits(dst)))
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode  %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVADDI32(state emu.InstEmuState) {
	inst := state.Inst()

	if inst.IsSdwa {
		u.runVADDI32SDWA(state)
	} else {
		u.runVADDI32Regular(state)
	}
}

func (u *ALU) runVADDI32SDWA(state emu.InstEmuState) {
	inst := state.Inst()
	exec := state.EXEC()
	var vcc uint64

	for i := 0; i < 64; i++ {
		if exec&(1<<uint(i)) == 0 {
			continue
		}

		src0 := u.sdwaSrcSelect(uint32(state.ReadOperand(inst.Src0, i)), inst.Src0Sel)
		src1 := u.sdwaSrcSelect(uint32(state.ReadOperand(inst.Src1, i)), inst.Src1Sel)
		sum := src0 + src1
		if uint64(src0)+uint64(src1) > 0xffffffff {
			vcc |= 1 << uint32(i)
		}
		dst := u.sdwaDstSelect(uint32(state.ReadOperand(inst.Dst, i)), sum,
			inst.DstSel, inst.DstUnused)
		state.WriteOperand(inst.Dst, i, uint64(dst))
	}
	if vop2WritesVCC(inst.Opcode) {
		state.SetVCC(vcc)
	}
}

func (u *ALU) runVADDI32Regular(state emu.InstEmuState) {
	inst := state.Inst()
	exec := state.EXEC()
	var vcc uint64

	for i := 0; i < 64; i++ {
		if exec&(1<<uint(i)) == 0 {
			continue
		}

		src0 := uint32(state.ReadOperand(inst.Src0, i))
		src1 := uint32(state.ReadOperand(inst.Src1, i))

		if uint64(src0)+uint64(src1) > 0xffffffff {
			vcc |= 1 << uint32(i)
		}

		state.WriteOperand(inst.Dst, i, uint64(src0+src1))
	}
	if vop2WritesVCC(inst.Opcode) {
		state.SetVCC(vcc)
	}
}

func (u *ALU) runVSUBI32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		var vcc uint64
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			src0 := uint32(state.ReadOperand(inst.Src0, i))
			src1 := uint32(state.ReadOperand(inst.Src1, i))

			if src0 < src1 {
				vcc |= 1 << uint32(i)
			}

			state.WriteOperand(inst.Dst, i, uint64(src0-src1))
		}
		if vop2WritesVCC(inst.Opcode) {
			state.SetVCC(vcc)
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode  %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVSUBREVI32(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		var vcc uint64
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			src0 := uint32(state.ReadOperand(inst.Src0, i))
			src1 := uint32(state.ReadOperand(inst.Src1, i))

			if src0 > src1 {
				vcc |= 1 << uint32(i)
			}

			state.WriteOperand(inst.Dst, i, uint64(src1-src0))
		}
		if vop2WritesVCC(inst.Opcode) {
			state.SetVCC(vcc)
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode  %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVADDCU32(state emu.InstEmuState) {
	inst := state.Inst()

	if !inst.IsSdwa {
		exec := state.EXEC()
		oldVCC := state.VCC()
		var newVCC uint64
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			carry := (oldVCC & (1 << uint(i))) >> uint(i)

			src0 := state.ReadOperand(inst.Src0, i)
			src1 := state.ReadOperand(inst.Src1, i)

			if src0 > math.MaxUint32-carry-src1 {
				newVCC |= 1 << uint32(i)
			}

			state.WriteOperand(inst.Dst, i, src0+src1+carry)
		}
		state.SetVCC(newVCC)
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode  %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVSUBBU32(state emu.InstEmuState) {
	inst := state.Inst()

	if !inst.IsSdwa {
		exec := state.EXEC()
		oldVCC := state.VCC()
		var newVCC uint64
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			borrow := (oldVCC & (1 << uint(i))) >> uint(i)
			src0 := state.ReadOperand(inst.Src0, i)
			src1 := state.ReadOperand(inst.Src1, i)
			state.WriteOperand(inst.Dst, i, src0-src1-borrow)

			if src0 < src1+borrow {
				newVCC |= 1 << uint(i)
			}
		}
		state.SetVCC(newVCC)
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode  %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVSUBBREVU32(state emu.InstEmuState) {
	inst := state.Inst()

	if !inst.IsSdwa {
		exec := state.EXEC()
		oldVCC := state.VCC()
		var newVCC uint64
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}

			borrow := (oldVCC & (1 << uint(i))) >> uint(i)
			src0 := state.ReadOperand(inst.Src0, i)
			src1 := state.ReadOperand(inst.Src1, i)

			if src1 < src0+borrow {
				newVCC |= 1 << uint32(i)
			}

			state.WriteOperand(inst.Dst, i, src1-src0-borrow)
		}
		state.SetVCC(newVCC)
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode  %d not implemented \n", inst.Opcode)
	}
}

func (u *ALU) runVLSHLREVB16(state emu.InstEmuState) {
	inst := state.Inst()
	if !inst.IsSdwa {
		exec := state.EXEC()
		for i := 0; i < 64; i++ {
			if exec&(1<<uint(i)) == 0 {
				continue
			}
			src0 := uint16(state.ReadOperand(inst.Src0, i))
			src1 := uint16(state.ReadOperand(inst.Src1, i))
			dst := src1 << (src0 & 0xF)
			state.WriteOperand(inst.Dst, i, uint64(dst))
		}
	} else {
		log.Panicf("SDWA for VOP2 instruction opcode %d not implemented\n", inst.Opcode)
	}
}

// GFX9+: D.u16 = S0.u16 + S1.u16 (low 16 bits; high bits cleared).
func (u *ALU) runVADDU16(state emu.InstEmuState) {
	inst := state.Inst()
	if inst.IsSdwa {
		log.Panicf("SDWA for VOP2 instruction opcode %d not implemented\n", inst.Opcode)
	}
	exec := state.EXEC()
	for i := 0; i < 64; i++ {
		if exec&(1<<uint(i)) == 0 {
			continue
		}
		src0 := uint16(state.ReadOperand(inst.Src0, i))
		src1 := uint16(state.ReadOperand(inst.Src1, i))
		state.WriteOperand(inst.Dst, i, uint64(src0+src1))
	}
}
