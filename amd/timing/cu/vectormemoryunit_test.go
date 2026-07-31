package cu

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sarchlab/akita/v5/mem"
	"github.com/sarchlab/akita/v5/mem/memprotocol"
	"github.com/sarchlab/akita/v5/messaging"
	"github.com/sarchlab/akita/v5/modeling"
	"github.com/sarchlab/akita/v5/queueing"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
	"github.com/sarchlab/mgpusim/v5/amd/kernels"
	"github.com/sarchlab/mgpusim/v5/amd/timing/wavefront"
)

type fakeCoalescer struct {
	toReturn []VectorMemAccessInfo
	received []*wavefront.Wavefront
}

func (c *fakeCoalescer) generateMemTransactions(
	wf *wavefront.Wavefront,
) []VectorMemAccessInfo {
	c.received = append(c.received, wf)
	return c.toReturn
}

var _ = Describe("Vector Memory Unit", func() {

	var (
		cu          *ComputeUnit
		coalescer   *fakeCoalescer
		vecMemUnit  *VectorMemoryUnit
		toVectorMem *fakePort
	)

	BeforeEach(func() {
		engine := newFakeEngine()
		comp := modeling.NewBuilder[Spec, State, Resources]().
			WithEngine(engine).
			WithFreq(1 * timing.GHz).
			WithSpec(DefaultSpec()).
			WithResources(Resources{
				VectorMemModules: &mem.SinglePortMapper{Port: "VectorMem"},
			}).
			Build("CU")

		cu = &ComputeUnit{
			comp:   comp,
			engine: engine,
			wftime: make(map[uint64]timing.VTimeInPicoSec),
		}
		cu.InFlightVectorMemAccessLimit = 128

		coalescer = new(fakeCoalescer)
		vecMemUnit = NewVectorMemoryUnit(cu, coalescer)
		toVectorMem = newFakePort("CU.VectorMem")
		cu.ToVectorMem = toVectorMem

		vecMemUnit.instructionPipeline =
			queueing.NewPipeline[vectorMemInst](1, 6)
		vecMemUnit.postInstructionPipelineBuffer =
			queueing.NewBuffer[vectorMemInst]("CU.PostInstBuf", 16)
		vecMemUnit.transactionPipeline =
			queueing.NewPipeline[VectorMemAccessInfo](2, 10)
		vecMemUnit.postTransactionPipelineBuffer =
			queueing.NewBuffer[VectorMemAccessInfo]("CU.PostTransBuf", 8)
	})

	It("calculates read and write penalties from cache-line utilization", func() {
		vecMemUnit.maxCoalescingPenalty = 16
		vecMemUnit.maxWriteCoalescingPenalty = 32

		readTxn := VectorMemAccessInfo{
			Read: &memprotocol.ReadReq{AccessByteSize: 64},
			laneInfo: []vectorMemAccessLaneInfo{
				{addrOffsetInCacheLine: 0},
				{addrOffsetInCacheLine: 4},
			},
		}
		Expect(vecMemUnit.computeCoalescingPenalty(readTxn)).To(Equal(14))

		writeTxn := VectorMemAccessInfo{
			Write: &memprotocol.WriteReq{DirtyMask: make([]bool, 64)},
		}
		for i := 0; i < 16; i++ {
			writeTxn.Write.DirtyMask[i] = true
		}
		Expect(vecMemUnit.computeCoalescingPenalty(writeTxn)).To(Equal(24))
	})

	It("adds a penalty only for multi-lane reads using at most two words", func() {
		vecMemUnit.maxCoalescingPenalty = 16
		vecMemUnit.maxSparseReadCoalescingPenalty = 32

		twoWordTxn := VectorMemAccessInfo{
			Read: &memprotocol.ReadReq{AccessByteSize: 64},
			laneInfo: []vectorMemAccessLaneInfo{
				{addrOffsetInCacheLine: 0},
				{addrOffsetInCacheLine: 4},
			},
		}
		Expect(vecMemUnit.computeCoalescingPenalty(twoWordTxn)).To(Equal(42))

		fourWordTxn := VectorMemAccessInfo{
			Read: &memprotocol.ReadReq{AccessByteSize: 64},
			laneInfo: []vectorMemAccessLaneInfo{
				{addrOffsetInCacheLine: 0},
				{addrOffsetInCacheLine: 4},
				{addrOffsetInCacheLine: 8},
				{addrOffsetInCacheLine: 12},
			},
		}
		Expect(vecMemUnit.computeCoalescingPenalty(fourWordTxn)).To(Equal(12))
	})

	It("uses ordinary utilization cost for MUBUF scratch writes", func() {
		vecMemUnit.maxCoalescingPenalty = 16
		vecMemUnit.maxWriteCoalescingPenalty = 32
		inst := wavefront.NewInst(insts.NewInst())
		inst.FormatType = insts.MUBUF
		writeTxn := VectorMemAccessInfo{
			Write: &memprotocol.WriteReq{DirtyMask: make([]bool, 64)},
			Inst:  inst,
		}
		for i := 0; i < 16; i++ {
			writeTxn.Write.DirtyMask[i] = true
		}

		Expect(vecMemUnit.computeCoalescingPenalty(writeTxn)).To(Equal(12))
	})

	It("does not penalize same-word broadcasts across lanes", func() {
		vecMemUnit.maxCoalescingPenalty = 16
		readTxn := VectorMemAccessInfo{
			Read: &memprotocol.ReadReq{AccessByteSize: 64},
			laneInfo: []vectorMemAccessLaneInfo{
				{laneID: 0, addrOffsetInCacheLine: 4},
				{laneID: 1, addrOffsetInCacheLine: 4},
				{laneID: 2, addrOffsetInCacheLine: 4},
			},
		}

		Expect(vecMemUnit.computeCoalescingPenalty(readTxn)).To(Equal(0))
	})

	It("does not double-count a single-lane cache-line transaction", func() {
		vecMemUnit.maxCoalescingPenalty = 16
		readTxn := VectorMemAccessInfo{
			Read: &memprotocol.ReadReq{AccessByteSize: 64},
			laneInfo: []vectorMemAccessLaneInfo{
				{laneID: 7, addrOffsetInCacheLine: 12},
			},
		}

		Expect(vecMemUnit.computeCoalescingPenalty(readTxn)).To(Equal(0))
	})

	It("penalizes non-local but not adjacent write cache lines", func() {
		vecMemUnit.maxWideWriteStridePenalty = 20
		inst := wavefront.NewInst(insts.NewInst())
		inst.FormatType = insts.FLAT
		inst.Opcode = 31
		fullLine := func(address uint64) VectorMemAccessInfo {
			mask := make([]bool, 64)
			for i := range mask {
				mask[i] = true
			}
			return VectorMemAccessInfo{
				Write: &memprotocol.WriteReq{
					Address:   address,
					DirtyMask: mask,
				},
				Inst: inst,
			}
		}

		Expect(vecMemUnit.computeCoalescingPenalty(
			fullLine(0x1000))).To(Equal(0))
		Expect(vecMemUnit.computeCoalescingPenalty(
			fullLine(0x1040))).To(Equal(0))
		Expect(vecMemUnit.computeCoalescingPenalty(
			fullLine(0x1800))).To(Equal(20))
	})

	It("compares aligned cache lines for write locality", func() {
		Expect(cacheLinesAreLocal(0x1004, 0x107c, 64)).To(BeTrue())
		Expect(cacheLinesAreLocal(0x103c, 0x1804, 64)).To(BeFalse())
	})

	It("only recognizes opcode 31 stores in vector-memory formats", func() {
		inst := wavefront.NewInst(insts.NewInst())
		inst.FormatType = insts.SOP2
		inst.Opcode = 31

		Expect(isWideVectorWrite(VectorMemAccessInfo{Inst: inst})).To(BeFalse())
	})

	It("does not apply the wide-store locality cost to dword stores", func() {
		vecMemUnit.maxWideWriteStridePenalty = 20
		inst := wavefront.NewInst(insts.NewInst())
		inst.Opcode = 28
		fullMask := make([]bool, 64)
		for i := range fullMask {
			fullMask[i] = true
		}

		for _, address := range []uint64{0x1000, 0x1800} {
			txn := VectorMemAccessInfo{
				Write: &memprotocol.WriteReq{
					Address:   address,
					DirtyMask: fullMask,
				},
				Inst: inst,
			}
			Expect(vecMemUnit.computeCoalescingPenalty(txn)).To(Equal(0))
		}
	})

	It("keeps ticking while a coalescing penalty drains", func() {
		vecMemUnit.coalescingStallRemaining = 2

		Expect(vecMemUnit.insertTransactionToPipeline()).To(BeTrue())
		Expect(vecMemUnit.coalescingStallRemaining).To(Equal(1))
	})

	It("should allow accepting wavefront", func() {
		Expect(vecMemUnit.CanAcceptWave()).To(BeTrue())
	})

	It("should not allow accepting wavefront if the instruction pipeline "+
		"entry stage is occupied", func() {
		vecMemUnit.instructionPipeline.Accept(vectorMemInst{})
		Expect(vecMemUnit.CanAcceptWave()).To(BeFalse())
	})

	It("should accept wave", func() {
		wave := new(wavefront.Wavefront)

		vecMemUnit.AcceptWave(wave)

		Expect(vecMemUnit.numInstInFlight).To(Equal(uint64(1)))
		Expect(vecMemUnit.instructionPipeline.Stages()).To(HaveLen(1))
	})

	It("should run flat_load_dword", func() {
		kernelWave := kernels.NewWavefront()
		wave := wavefront.NewWavefront(kernelWave)
		inst := wavefront.NewInst(insts.NewInst())
		inst.Format = insts.FormatTable[insts.FLAT]
		inst.Opcode = 20
		inst.Dst = insts.NewVRegOperand(0, 0, 1)
		wave.SetDynamicInst(inst)

		transactions := make([]VectorMemAccessInfo, 4)
		for i := 0; i < 4; i++ {
			read := &memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
				Address:        0x100,
				AccessByteSize: 4,
			}
			transactions[i].Read = read
		}
		coalescer.toReturn = transactions

		vecMemUnit.postInstructionPipelineBuffer.PushTyped(
			vectorMemInst{wavefront: wave})
		vecMemUnit.numInstInFlight = 1

		madeProgress := vecMemUnit.instToTransaction()

		Expect(madeProgress).To(BeTrue())
		Expect(coalescer.received).To(ContainElement(wave))
		Expect(wave.State).To(Equal(wavefront.WfReady))
		Expect(wave.OutstandingVectorMemAccess).To(Equal(1))
		Expect(wave.OutstandingScalarMemAccess).To(Equal(1))
		Expect(cu.InFlightVectorMemAccess).To(HaveLen(4))
		Expect(cu.InFlightVectorMemAccess[3].Read.CanWaitForCoalesce).
			To(BeFalse())
		Expect(vecMemUnit.transactionsWaiting).To(HaveLen(4))
		Expect(vecMemUnit.postInstructionPipelineBuffer.Size()).To(Equal(0))
		Expect(vecMemUnit.numInstInFlight).To(Equal(uint64(0)))
	})

	It("should run flat_store_dword", func() {
		kernelWave := kernels.NewWavefront()
		wave := wavefront.NewWavefront(kernelWave)
		inst := wavefront.NewInst(insts.NewInst())
		inst.Format = insts.FormatTable[insts.FLAT]
		inst.Opcode = 28
		inst.Dst = insts.NewVRegOperand(0, 0, 1)
		wave.SetDynamicInst(inst)

		transactions := make([]VectorMemAccessInfo, 4)
		for i := 0; i < 4; i++ {
			write := &memprotocol.WriteReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
				Address: 0x100,
			}
			transactions[i].Write = write
		}
		coalescer.toReturn = transactions

		vecMemUnit.postInstructionPipelineBuffer.PushTyped(
			vectorMemInst{wavefront: wave})
		vecMemUnit.numInstInFlight = 1

		madeProgress := vecMemUnit.instToTransaction()

		Expect(madeProgress).To(BeTrue())
		Expect(wave.State).To(Equal(wavefront.WfReady))
		Expect(wave.OutstandingVectorMemAccess).To(Equal(1))
		Expect(wave.OutstandingScalarMemAccess).To(Equal(1))
		Expect(cu.InFlightVectorMemAccess).To(HaveLen(4))
		Expect(cu.InFlightVectorMemAccess[3].Write.CanWaitForCoalesce).
			To(BeFalse())
		Expect(vecMemUnit.transactionsWaiting).To(HaveLen(4))
	})

	It("should add transactions to pipeline", func() {
		transactions := make([]VectorMemAccessInfo, 4)
		for i := 0; i < 4; i++ {
			write := &memprotocol.WriteReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
				Address: 0x100,
			}
			transactions[i].Write = write
		}
		vecMemUnit.transactionsWaiting = transactions

		// The transaction pipeline has width 2, so only 2 transactions can
		// enter stage 0 in one cycle.
		madeProgress := vecMemUnit.instToTransaction()

		Expect(madeProgress).To(BeTrue())
		Expect(vecMemUnit.transactionsWaiting).To(HaveLen(2))
		Expect(vecMemUnit.transactionPipeline.Stages()).To(HaveLen(2))
	})

	It("should send memory access requests", func() {
		inst := wavefront.NewInst(nil)
		loadReq := &memprotocol.ReadReq{
			MsgMeta: messaging.MsgMeta{
				ID:  timing.GetIDGenerator().Generate(),
				Src: cu.ToVectorMem.AsRemote(),
				Dst: "VectorMem",
			},
			Address:        0,
			AccessByteSize: 4,
		}
		trans := VectorMemAccessInfo{
			Read: loadReq,
			Inst: inst,
		}
		vecMemUnit.numTransactionInFlight = 1

		vecMemUnit.postTransactionPipelineBuffer.PushTyped(trans)

		vecMemUnit.sendRequest()

		Expect(toVectorMem.sent).To(HaveLen(1))
		Expect(toVectorMem.sent[0]).To(Equal(*loadReq))
		Expect(vecMemUnit.numTransactionInFlight).To(Equal(uint64(0)))
		Expect(vecMemUnit.postTransactionPipelineBuffer.Size()).To(Equal(0))
	})

	It("should flush the vector memory unit", func() {
		vecMemUnit.numInstInFlight = 1
		vecMemUnit.numTransactionInFlight = 1
		vecMemUnit.transactionsWaiting = append(vecMemUnit.transactionsWaiting,
			VectorMemAccessInfo{})
		vecMemUnit.instructionPipeline.Accept(vectorMemInst{})
		vecMemUnit.postInstructionPipelineBuffer.PushTyped(vectorMemInst{})

		vecMemUnit.Flush()

		Expect(vecMemUnit.numInstInFlight).To(Equal(uint64(0)))
		Expect(vecMemUnit.numTransactionInFlight).To(Equal(uint64(0)))
		Expect(vecMemUnit.transactionsWaiting).To(BeEmpty())
		Expect(vecMemUnit.instructionPipeline.Stages()).To(BeEmpty())
		Expect(vecMemUnit.postInstructionPipelineBuffer.Size()).To(Equal(0))
	})
})
