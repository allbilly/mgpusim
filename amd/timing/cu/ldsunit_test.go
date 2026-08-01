package cu

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
	"github.com/sarchlab/mgpusim/v5/amd/protocol"
	"github.com/sarchlab/mgpusim/v5/amd/timing/wavefront"
)

var _ = Describe("LDS Unit", func() {

	var (
		cu  *ComputeUnit
		bu  *LDSUnit
		alu *mockALU
	)

	BeforeEach(func() {
		cu = newTestComputeUnit("CU", nil)
		alu = new(mockALU)
		bu = NewLDSUnit(cu, alu)
	})

	It("should allow accepting wavefront", func() {
		// wave := new(Wavefront)
		bu.toRead = nil
		Expect(bu.CanAcceptWave()).To(BeTrue())
	})

	It("should not allow accepting wavefront is the read stage buffer is occupied", func() {
		bu.toRead = new(wavefront.Wavefront)
		Expect(bu.CanAcceptWave()).To(BeFalse())
	})

	It("should accept wave", func() {
		wave := new(wavefront.Wavefront)
		bu.AcceptWave(wave)
		Expect(bu.toRead).To(BeIdenticalTo(wave))
	})

	It("should run", func() {
		wave2 := new(wavefront.Wavefront)
		wave2.WG = wavefront.NewWorkGroup(nil, protocol.MapWGReq{})
		wave2.WG.LDS = make([]byte, 0)
		inst := wavefront.NewInst(insts.NewInst())
		inst.FormatType = insts.DS
		inst.Opcode = 0
		inst.Addr = insts.NewVRegOperand(0, 0, 1)
		inst.Data = insts.NewVRegOperand(2, 2, 2)
		inst.Data1 = insts.NewVRegOperand(4, 4, 2)
		inst.ByteSize = 4
		wave2.SetDynamicInst(inst)
		wave2.SetPC(0x13C)
		wave2.InstBuffer = make([]byte, 256)
		wave2.InstBufferStartPC = 0x100

		wave2.State = wavefront.WfRunning

		bu.toRead = wave2

		bu.Run()

		Expect(bu.toRead).To(BeNil())
		Expect(bu.inFlight).To(HaveLen(1))
		Expect(alu.wfExecuted).To(BeIdenticalTo(wave2))

		// Run 14 cycles to drain the default result latency.
		for i := 0; i < 13; i++ {
			bu.Run()
		}
		Expect(bu.inFlight).To(HaveLen(1))
		bu.Run()
		Expect(bu.inFlight).To(BeEmpty())
		Expect(wave2.State).To(Equal(wavefront.WfReady))
		Expect(wave2.PC()).To(Equal(uint64(0x140)))
		Expect(wave2.InstBuffer).To(HaveLen(192))
	})

	It("should pipeline independent LDS instructions", func() {
		spec := DefaultSpec()
		spec.LDSIssueInterval = 4
		spec.LDSMaxInFlight = 4
		cu = newTestComputeUnitWithSpec("PipelinedCU", nil, spec)
		bu = NewLDSUnit(cu, alu)

		first := new(wavefront.Wavefront)
		first.WG = wavefront.NewWorkGroup(nil, protocol.MapWGReq{})
		first.WG.LDS = make([]byte, 0)
		bu.toRead = first
		bu.Run()

		second := new(wavefront.Wavefront)
		second.WG = first.WG
		bu.toRead = second
		for i := 0; i < 4; i++ {
			bu.Run()
		}
		Expect(bu.inFlight).To(HaveLen(2))
	})

	It("adds B128 service cycles to issue occupancy and result latency", func() {
		spec := DefaultSpec()
		spec.LDSPipelineLatency = 14
		spec.LDSIssueInterval = 4
		spec.LDSMaxInFlight = 4
		spec.LDSB128ServiceExtraCycles = 7
		cu = newTestComputeUnitWithSpec("B128CU", nil, spec)
		bu = NewLDSUnit(cu, alu)

		wave := new(wavefront.Wavefront)
		wave.WG = wavefront.NewWorkGroup(nil, protocol.MapWGReq{})
		wave.WG.LDS = make([]byte, 1024)
		inst := wavefront.NewInst(insts.NewInst())
		inst.FormatType = insts.DS
		inst.Opcode = 255 // DS_READ_B128
		inst.Addr = insts.NewVRegOperand(0, 0, 1)
		inst.Dst = insts.NewVRegOperand(4, 7, 4)
		wave.SetDynamicInst(inst)
		bu.toRead = wave

		bu.Run()

		Expect(bu.issueIntervalLeft).To(Equal(11))
		Expect(bu.inFlight).To(HaveLen(1))
		Expect(bu.inFlight[0].cyclesLeft).To(Equal(21))
	})

	It("should flush the LDS", func() {

		wave1 := new(wavefront.Wavefront)
		wave2 := new(wavefront.Wavefront)
		wave2.WG = wavefront.NewWorkGroup(nil, protocol.MapWGReq{})
		wave2.WG.LDS = make([]byte, 0)
		wave3 := new(wavefront.Wavefront)
		inst := wavefront.NewInst(insts.NewInst())
		inst.FormatType = insts.DS
		inst.Opcode = 0
		inst.Addr = insts.NewVRegOperand(0, 0, 1)
		inst.Data = insts.NewVRegOperand(2, 2, 2)
		inst.Data1 = insts.NewVRegOperand(4, 4, 2)
		inst.ByteSize = 4
		wave3.SetDynamicInst(inst)
		wave3.SetPC(0x13C)
		wave3.InstBuffer = make([]byte, 256)
		wave3.InstBufferStartPC = 0x100

		wave3.State = wavefront.WfRunning

		bu.toRead = wave1
		bu.inFlight = []ldsPipelineEntry{
			{wave: wave2, cyclesLeft: 5},
			{wave: wave3, cyclesLeft: 3},
		}
		bu.issueIntervalLeft = 2

		bu.Flush()

		Expect(bu.toRead).To(BeNil())
		Expect(bu.inFlight).To(BeEmpty())
		Expect(bu.issueIntervalLeft).To(Equal(0))

	})
})
