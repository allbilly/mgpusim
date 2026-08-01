package cu

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
	"github.com/sarchlab/mgpusim/v5/amd/timing/wavefront"
)

var _ = Describe("Default Coalescer", func() {
	var (
		wf          *wavefront.Wavefront
		c           defaultCoalescer
		regAccessor *mockRegFileAccessor
	)

	BeforeEach(func() {
		wf = wavefront.NewWavefront(nil)
		c = defaultCoalescer{
			log2CacheLineSize: 6,
		}
		regAccessor = newMockRegFileAccessor()
		wf.RegAccessor = regAccessor
	})

	It("should coalesce to a single cacheline", func() {
		inst := insts.NewInst()
		inst.FormatType = insts.FLAT
		inst.Opcode = 20 // flat_load_dword
		inst.Dst = insts.NewVRegOperand(0, 0, 1)
		inst.Addr = insts.NewVRegOperand(2, 2, 2)
		wf.SetDynamicInst(wavefront.NewInst(inst))
		wf.SetEXEC(0xffffffffffffffff)

		// Set all 64 lanes' ADDR (v2:v3) to 0x1000
		for i := 0; i < 64; i++ {
			addrReg := insts.VReg(2)
			regAccessor.setRegValue(addrReg, 2, i, wf.VRegOffset,
				insts.Uint64ToBytes(0x1000)[:8])
		}

		memTransactions := c.generateMemTransactions(wf)

		Expect(memTransactions).To(HaveLen(1))
		Expect(memTransactions[0].laneInfo).To(HaveLen(64))
	})

	It("should coalesce to multiple cachelines", func() {
		inst := insts.NewInst()
		inst.FormatType = insts.FLAT
		inst.Opcode = 20 // flat_load_dword
		inst.Dst = insts.NewVRegOperand(0, 0, 1)
		inst.Addr = insts.NewVRegOperand(2, 2, 2)
		wf.SetDynamicInst(wavefront.NewInst(inst))
		wf.SetEXEC(0xffffffffffffffff)

		for i := 0; i < 64; i++ {
			addrReg := insts.VReg(2)
			regAccessor.setRegValue(addrReg, 2, i, wf.VRegOffset,
				insts.Uint64ToBytes(uint64(0x1000 + i*4))[:8])
		}

		memTransactions := c.generateMemTransactions(wf)

		Expect(memTransactions).To(HaveLen(4))
		Expect(memTransactions[0].laneInfo).To(HaveLen(16))
		Expect(memTransactions[1].laneInfo).To(HaveLen(16))
		Expect(memTransactions[2].laneInfo).To(HaveLen(16))
		Expect(memTransactions[3].laneInfo).To(HaveLen(16))
	})

	It("should not generate cross-cache-line requests", func() {
		inst := insts.NewInst()
		inst.FormatType = insts.FLAT
		inst.Opcode = 21 // flat_load_dwordx2
		inst.Dst = insts.NewVRegOperand(0, 0, 1)
		inst.Addr = insts.NewVRegOperand(4, 4, 2)
		wf.SetDynamicInst(wavefront.NewInst(inst))
		wf.SetEXEC(0xffffffffffffffff)

		for i := 0; i < 64; i++ {
			addrReg := insts.VReg(4)
			regAccessor.setRegValue(addrReg, 2, i, wf.VRegOffset,
				insts.Uint64ToBytes(uint64(0x1004 + i*4))[:8])
		}

		memTransactions := c.generateMemTransactions(wf)

		Expect(memTransactions).To(HaveLen(5))
		Expect(memTransactions[0].laneInfo).To(HaveLen(29))
		Expect(memTransactions[1].laneInfo).To(HaveLen(32))
		Expect(memTransactions[2].laneInfo).To(HaveLen(32))
		Expect(memTransactions[3].laneInfo).To(HaveLen(32))
		Expect(memTransactions[4].laneInfo).To(HaveLen(3))
	})

	It("should coalesce store instructions", func() {
		inst := insts.NewInst()
		inst.FormatType = insts.FLAT
		inst.Opcode = 28 // flat_store_dword
		inst.Addr = insts.NewVRegOperand(2, 2, 2)
		inst.Data = insts.NewVRegOperand(4, 4, 1)
		wf.SetDynamicInst(wavefront.NewInst(inst))
		wf.SetEXEC(0xffffffffffffffff)

		for i := 0; i < 64; i++ {
			addrReg := insts.VReg(2)
			regAccessor.setRegValue(addrReg, 2, i, wf.VRegOffset,
				insts.Uint64ToBytes(uint64(0x1000 + i*4))[:8])

			dataReg := insts.VReg(4)
			regAccessor.setRegValue(dataReg, 1, i, wf.VRegOffset,
				insts.Uint32ToBytes(1))
		}

		memTransactions := c.generateMemTransactions(wf)

		Expect(memTransactions).To(HaveLen(4))
	})

	DescribeTable("counts only extra wide-load lane-dwords",
		func(opcode int, exec uint64, alias bool, expected int) {
			inst := insts.NewInst()
			inst.FormatType = insts.FLAT
			inst.Opcode = insts.Opcode(opcode)
			inst.Dst = insts.NewVRegOperand(0, 0, 1)
			inst.Addr = insts.NewVRegOperand(4, 4, 2)
			dynamicInst := wavefront.NewInst(inst)
			wf.SetDynamicInst(dynamicInst)
			wf.SetEXEC(exec)

			for lane := 0; lane < 64; lane++ {
				addr := uint64(0x1000)
				if !alias {
					addr += uint64(lane * 16)
				}
				regAccessor.setRegValue(
					insts.VReg(4), 2, lane, wf.VRegOffset,
					insts.Uint64ToBytes(addr)[:8],
				)
			}

			transactions := c.generateMemTransactions(wf)
			cu := &ComputeUnit{vmemWideLoadReturnLaneDwordsPerCycle: 1}
			for _, transaction := range transactions {
				cu.accumulateWideVMemReturn(
					dynamicInst.ID, transaction.laneInfo)
			}
			assembly := cu.vmemWideReturnAssemblies[dynamicInst.ID]
			extra := assembly.laneDwords - len(assembly.activeLanes)

			Expect(extra).To(Equal(expected))
		},
		Entry("dword full wave", 20, ^uint64(0), false, 0),
		Entry("dwordx2 full wave", 21, ^uint64(0), false, 64),
		Entry("dwordx4 full wave", 23, ^uint64(0), false, 192),
		Entry("dwordx4 full-wave alias", 23, ^uint64(0), true, 192),
		Entry("dwordx4 partial EXEC", 23, uint64(0xff), false, 24),
		Entry("ubyte full wave", 16, ^uint64(0), false, 0),
		Entry("ushort full wave", 18, ^uint64(0), false, 0),
	)

	DescribeTable("counts CU-wide return work from real coalescer entries",
		func(opcode int, exec uint64, alias, reverse bool, expected int) {
			inst := insts.NewInst()
			inst.FormatType = insts.FLAT
			inst.Opcode = insts.Opcode(opcode)
			inst.Dst = insts.NewVRegOperand(0, 0, 1)
			inst.Addr = insts.NewVRegOperand(4, 4, 2)
			dynamicInst := wavefront.NewInst(inst)
			wf.SetDynamicInst(dynamicInst)
			wf.SetEXEC(exec)

			for lane := 0; lane < 64; lane++ {
				addr := uint64(0x1000)
				if !alias {
					addr += uint64(lane * 16)
				}
				regAccessor.setRegValue(
					insts.VReg(4), 2, lane, wf.VRegOffset,
					insts.Uint64ToBytes(addr)[:8],
				)
			}

			transactions := c.generateMemTransactions(wf)
			cu := &ComputeUnit{vmemCUWideReturnUnitsPerCycle: 1}
			for i := range transactions {
				index := i
				if reverse {
					index = len(transactions) - 1 - i
				}
				cu.accumulateWideVMemReturn(
					dynamicInst.ID, transactions[index].laneInfo)
			}

			Expect(countCUWideReturnWork(
				cu.vmemWideReturnAssemblies[dynamicInst.ID],
			)).To(Equal(expected))
		},
		Entry("dword full wave", 20, ^uint64(0), false, false, 0),
		Entry("dwordx2 full wave", 21, ^uint64(0), false, false, 64),
		Entry("dwordx3 full wave", 22, ^uint64(0), false, false, 86),
		Entry("dwordx4 full wave", 23, ^uint64(0), false, false, 96),
		Entry("dwordx4 alias", 23, ^uint64(0), true, false, 96),
		Entry("dwordx4 partial EXEC", 23, uint64(0xff), false, false, 12),
		Entry("dwordx4 split reverse return", 23, ^uint64(0), false, true, 96),
		Entry("ubyte full wave", 16, ^uint64(0), false, false, 0),
		Entry("ushort full wave", 18, ^uint64(0), false, false, 0),
	)
})
