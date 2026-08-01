package cu

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sarchlab/akita/v5/mem/memprotocol"
	"github.com/sarchlab/akita/v5/messaging"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
	"github.com/sarchlab/mgpusim/v5/amd/kernels"
	"github.com/sarchlab/mgpusim/v5/amd/protocol"
	"github.com/sarchlab/mgpusim/v5/amd/timing/wavefront"
)

type mockScheduler struct {
}

func (m *mockScheduler) Run() bool {
	return true
}

func (m *mockScheduler) Pause() {
}

func (m *mockScheduler) Resume() {
}

func (m *mockScheduler) Flush() {
}

type mockDecoder struct {
	Inst *insts.Inst
}

func (d *mockDecoder) Decode(buf []byte) (*insts.Inst, error) {
	return d.Inst, nil
}

type fakeWfDispatcher struct {
	dispatched []protocol.WfDispatchLocation
}

func (d *fakeWfDispatcher) DispatchWf(
	wf *wavefront.Wavefront,
	location protocol.WfDispatchLocation,
) {
	d.dispatched = append(d.dispatched, location)
}

func exampleGrid() *kernels.Grid {
	grid := kernels.NewGrid()

	grid.CodeObject = &insts.KernelCodeObject{
		KernelCodeObjectMeta: &insts.KernelCodeObjectMeta{},
	}

	packet := new(kernels.HsaKernelDispatchPacket)
	grid.Packet = packet

	wg := kernels.NewWorkGroup()
	wg.Packet = packet
	wg.CodeObject = grid.CodeObject
	grid.WorkGroups = append(grid.WorkGroups, wg)

	wf := kernels.NewWavefront()
	wf.WG = wg
	wg.Wavefronts = append(wg.Wavefronts, wf)

	return grid
}

func makePendingVMemWave(
	simdID int,
	work ...int,
) (*wavefront.Wavefront, []pendingVMemLoadRetirement) {
	wf := wavefront.NewWavefront(kernels.NewWavefront())
	wf.SIMDID = simdID
	wf.OutstandingVectorMemAccess = len(work)
	wf.OutstandingScalarMemAccess = len(work)
	wf.InFlightInsts = len(work)

	retirements := make([]pendingVMemLoadRetirement, 0, len(work))
	for _, remaining := range work {
		inst := wavefront.NewInst(insts.NewInst())
		inst.FormatType = insts.FLAT
		retirements = append(retirements, pendingVMemLoadRetirement{
			wf:              wf,
			inst:            inst,
			remainingCycles: remaining,
		})
	}

	return wf, retirements
}

var _ = DescribeTable(
	"counts duplicate vector-memory return lane-dwords",
	func(alias, width int) {
		lanes := make([]vectorMemAccessLaneInfo, 0, 64*width)
		uniqueLanes := 64 / alias
		for laneID := 0; laneID < 64; laneID++ {
			for dword := 0; dword < width; dword++ {
				lanes = append(lanes, vectorMemAccessLaneInfo{
					laneID: laneID,
					addrOffsetInCacheLine: uint64(
						4 * ((laneID%uniqueLanes)*width + dword),
					),
				})
			}
		}

		want := 64 * width * (alias - 1) / alias
		Expect(countDuplicateVMemLaneDwords(lanes)).To(Equal(want))
	},
	Entry("alias 1 dword", 1, 1),
	Entry("alias 2 dword", 2, 1),
	Entry("alias 4 dword", 4, 1),
	Entry("alias 8 dword", 8, 1),
	Entry("alias 1 dwordx2", 1, 2),
	Entry("alias 2 dwordx2", 2, 2),
	Entry("alias 4 dwordx2", 4, 2),
	Entry("alias 8 dwordx2", 8, 2),
	Entry("alias 1 dwordx4", 1, 4),
	Entry("alias 2 dwordx4", 2, 4),
	Entry("alias 4 dwordx4", 4, 4),
	Entry("alias 8 dwordx4", 8, 4),
)

var _ = DescribeTable(
	"counts all vector-memory return lane-dwords independent of aliasing",
	func(alias, width int) {
		lanes := make([]vectorMemAccessLaneInfo, 0, 64*width)
		uniqueLanes := 64 / alias
		for laneID := 0; laneID < 64; laneID++ {
			for dword := 0; dword < width; dword++ {
				lanes = append(lanes, vectorMemAccessLaneInfo{
					laneID:   laneID,
					regCount: 1,
					addrOffsetInCacheLine: uint64(
						4 * (laneID%uniqueLanes + dword),
					),
				})
			}
		}

		Expect(countTotalVMemLaneDwords(lanes)).To(Equal(64 * width))
	},
	Entry("alias 1 dword", 1, 1),
	Entry("alias 8 dword", 8, 1),
	Entry("alias 1 dwordx2", 1, 2),
	Entry("alias 8 dwordx2", 8, 2),
	Entry("alias 1 dwordx4", 1, 4),
	Entry("alias 8 dwordx4", 8, 4),
)

var _ = Describe("ComputeUnit", func() {
	var (
		cu               *ComputeUnit
		engine           *fakeEngine
		wfDispatcher     *fakeWfDispatcher
		decoder          *mockDecoder
		toInstMem        *fakePort
		toScalarMem      *fakePort
		toVectorMem      *fakePort
		toACE            *fakePort
		toCP             *fakePort
		branchUnit       *mockCUComponent
		vectorMemDecoder *mockCUComponent
		vectorMemUnit    *mockCUComponent
		scalarDecoder    *mockCUComponent
		vectorDecoder    *mockCUComponent
		ldsDecoder       *mockCUComponent
		scalarUnit       *mockCUComponent
		simdUnit         *mockCUComponent
		ldsUnit          *mockCUComponent

		grid *kernels.Grid

		scheduler *mockScheduler
	)

	BeforeEach(func() {
		engine = newFakeEngine()
		wfDispatcher = new(fakeWfDispatcher)
		decoder = new(mockDecoder)
		scheduler = new(mockScheduler)
		branchUnit = new(mockCUComponent)
		vectorMemDecoder = new(mockCUComponent)
		vectorMemUnit = new(mockCUComponent)
		scalarDecoder = new(mockCUComponent)
		vectorDecoder = new(mockCUComponent)
		ldsDecoder = new(mockCUComponent)
		scalarUnit = new(mockCUComponent)
		simdUnit = new(mockCUComponent)
		ldsUnit = new(mockCUComponent)

		cu = newTestComputeUnit("CU", engine)
		cu.WfDispatcher = wfDispatcher
		cu.Decoder = decoder
		cu.SRegFile = NewSimpleRegisterFile(1024, 0)
		cu.VRegFile = append(cu.VRegFile, NewSimpleRegisterFile(4096, 64))
		cu.Scheduler = scheduler

		cu.BranchUnit = branchUnit
		cu.VectorMemDecoder = vectorMemDecoder
		cu.VectorMemUnit = vectorMemUnit
		cu.ScalarDecoder = scalarDecoder
		cu.VectorDecoder = vectorDecoder
		cu.LDSDecoder = ldsDecoder
		cu.ScalarUnit = scalarUnit
		cu.SIMDUnit = append(cu.SIMDUnit, simdUnit)

		cu.LDSUnit = ldsUnit

		for i := 0; i < 4; i++ {
			cu.WfPools = append(cu.WfPools, NewWavefrontPool(10))
		}

		toInstMem = newFakePort("CU.InstMem")
		toACE = newFakePort("CU.Top")
		toScalarMem = newFakePort("CU.ScalarMem")
		toVectorMem = newFakePort("CU.VectorMem")
		toCP = newFakePort("CU.Ctrl")
		cu.ToInstMem = toInstMem
		cu.ToACE = toACE
		cu.ToScalarMem = toScalarMem
		cu.ToVectorMem = toVectorMem
		cu.ToCP = toCP

		cu.comp.State.InstMem = "InstMem"
		cu.comp.State.ScalarMem = "ScalarMem"

		grid = exampleGrid()
	})

	Context("when processing MapWGReq", func() {
		var (
			req protocol.MapWGReq
		)

		BeforeEach(func() {
			wg := grid.WorkGroups[0]
			wg.Wavefronts = make([]*kernels.Wavefront, 2)
			wg.Wavefronts[0] = kernels.NewWavefront()
			wg.Wavefronts[1] = kernels.NewWavefront()
			location1 := protocol.WfDispatchLocation{
				Wavefront:  wg.Wavefronts[0],
				SIMDID:     1,
				VGPROffset: 100,
				SGPROffset: 10,
				LDSOffset:  100,
			}
			location2 := protocol.WfDispatchLocation{
				Wavefront:  wg.Wavefronts[1],
				SIMDID:     2,
				VGPROffset: 200,
				SGPROffset: 200,
				LDSOffset:  200,
			}

			req = protocol.MapWGReq{
				MsgMeta: messaging.MsgMeta{
					ID:  timing.GetIDGenerator().Generate(),
					Dst: cu.ToACE.AsRemote(),
				},
				WorkGroup: wg,
				Wavefronts: []protocol.WfDispatchLocation{
					location1, location2,
				},
			}

			toACE.incoming = append(toACE.incoming, req)
		})

		It("should dispatch wavefront", func() {
			engine.now = 11

			cu.processInputFromACE()

			Expect(wfDispatcher.dispatched).To(HaveLen(2))
			Expect(wfDispatcher.dispatched[0]).To(Equal(req.Wavefronts[0]))
			Expect(wfDispatcher.dispatched[1]).To(Equal(req.Wavefronts[1]))
			Expect(cu.WfPools[1].wfs).To(HaveLen(1))
			Expect(cu.WfPools[2].wfs).To(HaveLen(1))
		})
	})

	Context("when handling DataReady from ToInstMem Port", func() {
		var (
			wf *wavefront.Wavefront
		)
		BeforeEach(func() {
			wf = wavefront.NewWavefront(kernels.NewWavefront())
			inst := wavefront.NewInst(nil)
			wf.SetDynamicInst(inst)
			wf.SetPC(0x1000)

			req := memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID:  timing.GetIDGenerator().Generate(),
					Src: cu.ToInstMem.AsRemote(),
					Dst: cu.comp.State.InstMem,
				},
				Address:        0x100,
				AccessByteSize: 64,
			}

			dataReady := memprotocol.DataReadyRsp{
				MsgMeta: messaging.MsgMeta{
					ID:    timing.GetIDGenerator().Generate(),
					Src:   cu.comp.State.InstMem,
					Dst:   cu.ToInstMem.AsRemote(),
					RspTo: req.ID,
				},
				Data: []byte{
					1, 2, 3, 4, 5, 6, 7, 8,
					1, 2, 3, 4, 5, 6, 7, 8,
					1, 2, 3, 4, 5, 6, 7, 8,
					1, 2, 3, 4, 5, 6, 7, 8,
					1, 2, 3, 4, 5, 6, 7, 8,
					1, 2, 3, 4, 5, 6, 7, 8,
					1, 2, 3, 4, 5, 6, 7, 8,
					1, 2, 3, 4, 5, 6, 7, 8,
				},
			}

			toInstMem.incoming = append(toInstMem.incoming, dataReady)

			info := new(InstFetchReqInfo)
			info.Wavefront = wf
			info.Req = req
			cu.InFlightInstFetch = append(cu.InFlightInstFetch, info)
		})

		It("should handle fetch return", func() {
			engine.now = 10

			madeProgress := cu.processInputFromInstMem()

			Expect(wf.LastFetchTime).To(Equal(timing.VTimeInPicoSec(10)))
			Expect(wf.PC()).To(Equal(uint64(0x1000)))
			Expect(cu.InFlightInstFetch).To(HaveLen(0))
			Expect(wf.InstBuffer).To(HaveLen(64))
			Expect(madeProgress).To(BeTrue())
		})
	})

	Context("should handle DataReady from ToScalarMem port", func() {
		var (
			wf *wavefront.Wavefront
		)

		BeforeEach(func() {
			rawWf := grid.WorkGroups[0].Wavefronts[0]
			wf = wavefront.NewWavefront(rawWf)
			wf.SRegOffset = 0
			wf.OutstandingScalarMemAccess = 1
		})

		It("should handle scalar data load return", func() {
			cu.vmemCUWideReturnUnitsPerCycle = 1
			read := memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID:  timing.GetIDGenerator().Generate(),
					Src: cu.ToScalarMem.AsRemote(),
				},
				Address:        0x100,
				AccessByteSize: 64,
			}

			info := new(ScalarMemAccessInfo)
			info.Inst = wavefront.NewInst(insts.NewInst())
			info.Wavefront = wf
			info.DstSGPR = insts.SReg(0)
			info.Req = read
			cu.InFlightScalarMemAccess = append(
				cu.InFlightScalarMemAccess, info)

			rsp := memprotocol.DataReadyRsp{
				MsgMeta: messaging.MsgMeta{
					ID:    timing.GetIDGenerator().Generate(),
					RspTo: read.ID,
				},
				Data: insts.Uint32ToBytes(32),
			}
			toScalarMem.incoming = append(toScalarMem.incoming, rsp)

			cu.processInputFromScalarMem()

			access := RegisterAccess{
				Reg:        insts.SReg(0),
				RegCount:   1,
				WaveOffset: 0,
				Data:       make([]byte, 4),
			}
			cu.SRegFile.Read(access)
			Expect(insts.BytesToUint32(access.Data)).To(Equal(uint32(32)))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(0))
			Expect(cu.InFlightScalarMemAccess).To(HaveLen(0))
		})

		It("does not retire lgkmcnt when the last-generated scalar request "+
			"returns before a sibling", func() {
			inst := wavefront.NewInst(insts.NewInst())
			lastGenerated := memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID:  timing.GetIDGenerator().Generate(),
					Src: cu.ToScalarMem.AsRemote(),
				},
				Address:            0x100,
				AccessByteSize:     4,
				CanWaitForCoalesce: false,
			}
			sibling := memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID:  timing.GetIDGenerator().Generate(),
					Src: cu.ToScalarMem.AsRemote(),
				},
				Address:            0x140,
				AccessByteSize:     4,
				CanWaitForCoalesce: true,
			}
			for _, read := range []memprotocol.ReadReq{lastGenerated, sibling} {
				cu.InFlightScalarMemAccess = append(
					cu.InFlightScalarMemAccess,
					&ScalarMemAccessInfo{
						Inst:      inst,
						Wavefront: wf,
						DstSGPR:   insts.SReg(0),
						Req:       read,
					},
				)
			}

			toScalarMem.incoming = append(
				toScalarMem.incoming,
				memprotocol.DataReadyRsp{
					MsgMeta: messaging.MsgMeta{
						ID:    timing.GetIDGenerator().Generate(),
						RspTo: lastGenerated.ID,
					},
					Data: insts.Uint32ToBytes(32),
				},
			)
			cu.processInputFromScalarMem()

			Expect(wf.OutstandingScalarMemAccess).To(Equal(1))
			Expect(cu.InFlightScalarMemAccess).To(HaveLen(1))

			toScalarMem.incoming = append(
				toScalarMem.incoming,
				memprotocol.DataReadyRsp{
					MsgMeta: messaging.MsgMeta{
						ID:    timing.GetIDGenerator().Generate(),
						RspTo: sibling.ID,
					},
					Data: insts.Uint32ToBytes(64),
				},
			)
			cu.processInputFromScalarMem()

			Expect(wf.OutstandingScalarMemAccess).To(Equal(0))
			Expect(cu.InFlightScalarMemAccess).To(BeEmpty())
		})
	})

	Context("should handle DataReady from ToVectorMem", func() {
		var (
			rawWf *kernels.Wavefront
			wf    *wavefront.Wavefront
			inst  *wavefront.Inst
			read  *memprotocol.ReadReq
			info  VectorMemAccessInfo
		)

		BeforeEach(func() {
			rawWf = grid.WorkGroups[0].Wavefronts[0]
			inst = wavefront.NewInst(insts.NewInst())
			inst.FormatType = insts.FLAT
			wf = wavefront.NewWavefront(rawWf)
			wf.SIMDID = 0
			wf.SetDynamicInst(inst)
			wf.VRegOffset = 0
			wf.OutstandingVectorMemAccess = 1
			wf.OutstandingScalarMemAccess = 1

			read = &memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
				Address:            0x100,
				AccessByteSize:     16,
				CanWaitForCoalesce: true,
			}

			info = VectorMemAccessInfo{}
			info.Read = read
			info.Wavefront = wf
			info.Inst = inst
			info.laneInfo = []vectorMemAccessLaneInfo{
				{0, insts.VReg(0), 1, 0},
				{1, insts.VReg(0), 1, 4},
				{2, insts.VReg(0), 1, 8},
				{3, insts.VReg(0), 1, 12},
			}
			cu.InFlightVectorMemAccess = append(
				cu.InFlightVectorMemAccess, info)

			dataReady := memprotocol.DataReadyRsp{
				MsgMeta: messaging.MsgMeta{
					ID:    timing.GetIDGenerator().Generate(),
					RspTo: read.ID,
				},
				Data: make([]byte, 16),
			}
			for i := 0; i < 4; i++ {
				copy(dataReady.Data[i*4:i*4+4],
					insts.Uint32ToBytes(uint32(i)))
			}
			toVectorMem.incoming = append(toVectorMem.incoming, dataReady)
		})

		It("should handle vector data load return, and the return is not "+
			"the last one for an instruction", func() {
			// The request generated last can return before an older sibling.
			read.CanWaitForCoalesce = false
			sibling := info
			sibling.Read = &memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
			}
			cu.InFlightVectorMemAccess = append(
				cu.InFlightVectorMemAccess, sibling)

			cu.processInputFromVectorMem()

			for i := 0; i < 4; i++ {
				access := RegisterAccess{}
				access.RegCount = 1
				access.WaveOffset = 0
				access.LaneID = i
				access.Reg = insts.VReg(0)
				access.Data = make([]byte, access.RegCount*4)
				cu.VRegFile[0].Read(access)
				Expect(insts.BytesToUint32(access.Data)).To(Equal(uint32(i)))
			}

			Expect(wf.OutstandingVectorMemAccess).To(Equal(1))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(1))
			Expect(cu.InFlightVectorMemAccess).To(HaveLen(1))
		})

		It("should handle vector data load return, and the return is the "+
			"last one for an instruction", func() {
			read.CanWaitForCoalesce = false

			cu.processInputFromVectorMem()

			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(0))
			Expect(cu.vmemReturnAssemblies).To(BeNil())
			Expect(cu.pendingVMemRetirements).To(BeEmpty())
			for i := 0; i < 4; i++ {
				access := RegisterAccess{}
				access.RegCount = 1
				access.WaveOffset = 0
				access.LaneID = i
				access.Reg = insts.VReg(0)
				access.Data = make([]byte, access.RegCount*4)
				cu.VRegFile[0].Read(access)
				Expect(insts.BytesToUint32(access.Data)).To(Equal(uint32(i)))
			}
		})

		It("retires an enabled load with no duplicate destinations immediately", func() {
			cu.vmemReturnFanoutLaneDwordsPerCycle = 1

			cu.processInputFromVectorMem()

			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(0))
			Expect(cu.pendingVMemRetirements).To(BeEmpty())
		})

		It("retires a dword load immediately under the wide-only model", func() {
			cu.vmemWideLoadReturnLaneDwordsPerCycle = 1

			cu.processInputFromVectorMem()

			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(0))
			Expect(cu.vmemWideReturnAssemblies).NotTo(HaveKey(inst.ID))
			Expect(cu.pendingVMemRetirements).To(BeEmpty())
		})

		It("retires a dword load immediately under the CU-wide model", func() {
			cu.vmemCUWideReturnUnitsPerCycle = 1

			cu.processInputFromVectorMem()

			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(0))
			Expect(cu.vmemWideReturnAssemblies).NotTo(HaveKey(inst.ID))
			Expect(cu.pendingVMemRetirements).To(BeEmpty())
		})

		It("aggregates sibling fanout and delays final retirement", func() {
			cu.vmemReturnFanoutLaneDwordsPerCycle = 2
			wf.InFlightInsts = 1
			cu.InFlightVectorMemAccess[0].laneInfo[1].addrOffsetInCacheLine = 0
			cu.InFlightVectorMemAccess[0].laneInfo[3].addrOffsetInCacheLine = 8

			sibling := info
			sibling.Read = &memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
			}
			sibling.laneInfo[1].addrOffsetInCacheLine = 0
			sibling.laneInfo[3].addrOffsetInCacheLine = 8
			cu.InFlightVectorMemAccess = append(
				cu.InFlightVectorMemAccess, sibling)

			cu.processInputFromVectorMem()

			Expect(cu.vmemReturnAssemblies[inst.ID]).To(Equal(2))
			Expect(wf.OutstandingVectorMemAccess).To(Equal(1))

			toVectorMem.incoming = append(
				toVectorMem.incoming,
				memprotocol.DataReadyRsp{
					MsgMeta: messaging.MsgMeta{
						ID:    timing.GetIDGenerator().Generate(),
						RspTo: sibling.Read.ID,
					},
					Data: make([]byte, 16),
				},
			)
			cu.processInputFromVectorMem()

			Expect(cu.vmemReturnAssemblies).NotTo(HaveKey(inst.ID))
			Expect(cu.pendingVMemRetirements).To(HaveLen(1))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(2))
			Expect(wf.OutstandingVectorMemAccess).To(Equal(1))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(1))
			waitInst := wavefront.NewInst(insts.NewInst())
			waitInst.Format = insts.FormatTable[insts.SOPP]
			waitInst.Opcode = 12
			waitInst.VMCNT = 0
			waitInst.LKGMCNT = 0
			wf.SetDynamicInst(waitInst)
			wf.State = wavefront.WfRunning
			wf.InFlightInsts++
			waitScheduler := NewScheduler(cu, nil, nil)
			waitScheduler.internalExecuting = []*wavefront.Wavefront{wf}

			Expect(waitScheduler.EvaluateInternalInst()).To(BeFalse())
			Expect(waitScheduler.internalExecuting).To(ContainElement(wf))

			Expect(cu.advanceVMemLoadRetirements()).To(BeTrue())
			Expect(wf.OutstandingVectorMemAccess).To(Equal(1))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(1))
			Expect(waitScheduler.EvaluateInternalInst()).To(BeFalse())
			Expect(waitScheduler.internalExecuting).To(ContainElement(wf))

			Expect(cu.advanceVMemLoadRetirements()).To(BeTrue())
			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(0))
			Expect(cu.pendingVMemRetirements).To(BeEmpty())
			Expect(waitScheduler.EvaluateInternalInst()).To(BeTrue())
			Expect(waitScheduler.internalExecuting).NotTo(ContainElement(wf))
			Expect(wf.State).To(Equal(wavefront.WfReady))
			Expect(wf.InFlightInsts).To(Equal(0))
		})

		It("aggregates all sibling lane-dwords and blocks waits until retirement", func() {
			cu.vmemLoadReturnLaneDwordsPerCycle = 3
			wf.InFlightInsts = 1

			sibling := info
			sibling.Read = &memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
			}
			cu.InFlightVectorMemAccess = append(
				cu.InFlightVectorMemAccess, sibling)

			cu.processInputFromVectorMem()

			Expect(cu.vmemReturnAssemblies[inst.ID]).To(Equal(4))
			Expect(wf.OutstandingVectorMemAccess).To(Equal(1))

			toVectorMem.incoming = append(
				toVectorMem.incoming,
				memprotocol.DataReadyRsp{
					MsgMeta: messaging.MsgMeta{
						ID:    timing.GetIDGenerator().Generate(),
						RspTo: sibling.Read.ID,
					},
					Data: make([]byte, 16),
				},
			)
			cu.processInputFromVectorMem()

			Expect(cu.vmemReturnAssemblies).NotTo(HaveKey(inst.ID))
			Expect(cu.pendingVMemRetirements).To(HaveLen(1))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(3))

			waitInst := wavefront.NewInst(insts.NewInst())
			waitInst.Format = insts.FormatTable[insts.SOPP]
			waitInst.Opcode = 12
			waitInst.VMCNT = 0
			waitInst.LKGMCNT = 0
			wf.SetDynamicInst(waitInst)
			wf.State = wavefront.WfRunning
			wf.InFlightInsts++
			waitScheduler := NewScheduler(cu, nil, nil)
			waitScheduler.internalExecuting = []*wavefront.Wavefront{wf}

			for remaining := 2; remaining >= 0; remaining-- {
				Expect(waitScheduler.EvaluateInternalInst()).To(BeFalse())
				Expect(cu.advanceVMemLoadRetirements()).To(BeTrue())
				if remaining > 0 {
					Expect(cu.pendingVMemRetirements[0].remainingCycles).
						To(Equal(remaining))
				}
			}

			Expect(cu.pendingVMemRetirements).To(BeEmpty())
			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(0))
			Expect(waitScheduler.EvaluateInternalInst()).To(BeTrue())
		})

		It("advances independent load retirements in parallel", func() {
			cu.vmemLoadReturnLaneDwordsPerCycle = 4
			otherInst := wavefront.NewInst(insts.NewInst())
			otherInst.FormatType = insts.FLAT
			otherWf := wavefront.NewWavefront(kernels.NewWavefront())
			for _, wave := range []*wavefront.Wavefront{wf, otherWf} {
				wave.OutstandingVectorMemAccess = 1
				wave.OutstandingScalarMemAccess = 1
				wave.InFlightInsts = 1
			}
			cu.pendingVMemRetirements = []pendingVMemLoadRetirement{
				{wf: wf, inst: inst, remainingCycles: 2},
				{wf: otherWf, inst: otherInst, remainingCycles: 2},
			}

			cu.advanceVMemLoadRetirements()

			Expect(cu.pendingVMemRetirements).To(HaveLen(2))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(1))
			Expect(cu.pendingVMemRetirements[1].remainingCycles).To(Equal(1))
			Expect(wf.OutstandingVectorMemAccess).To(Equal(1))
			Expect(otherWf.OutstandingVectorMemAccess).To(Equal(1))

			cu.advanceVMemLoadRetirements()

			Expect(cu.pendingVMemRetirements).To(BeEmpty())
			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(otherWf.OutstandingVectorMemAccess).To(Equal(0))
		})

		It("aggregates wide siblings, rounds up, and blocks waits", func() {
			cu.vmemWideLoadReturnLaneDwordsPerCycle = 3
			wf.InFlightInsts = 1

			sibling := info
			sibling.Read = &memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
			}
			for i := range sibling.laneInfo {
				sibling.laneInfo[i].reg = insts.VReg(1)
			}
			cu.InFlightVectorMemAccess = append(
				cu.InFlightVectorMemAccess, sibling)

			cu.processInputFromVectorMem()

			assembly := cu.vmemWideReturnAssemblies[inst.ID]
			Expect(assembly.laneDwords).To(Equal(4))
			Expect(assembly.activeLanes).To(HaveLen(4))
			Expect(cu.pendingVMemRetirements).To(BeEmpty())

			toVectorMem.incoming = append(
				toVectorMem.incoming,
				memprotocol.DataReadyRsp{
					MsgMeta: messaging.MsgMeta{
						ID:    timing.GetIDGenerator().Generate(),
						RspTo: sibling.Read.ID,
					},
					Data: make([]byte, 16),
				},
			)
			cu.processInputFromVectorMem()

			Expect(cu.vmemWideReturnAssemblies).NotTo(HaveKey(inst.ID))
			Expect(cu.pendingVMemRetirements).To(HaveLen(1))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(2))

			waitInst := wavefront.NewInst(insts.NewInst())
			waitInst.Format = insts.FormatTable[insts.SOPP]
			waitInst.Opcode = 12
			waitInst.VMCNT = 0
			waitInst.LKGMCNT = 0
			wf.SetDynamicInst(waitInst)
			wf.State = wavefront.WfRunning
			wf.InFlightInsts++
			waitScheduler := NewScheduler(cu, nil, nil)
			waitScheduler.internalExecuting = []*wavefront.Wavefront{wf}

			Expect(waitScheduler.EvaluateInternalInst()).To(BeFalse())
			Expect(cu.advanceVMemLoadRetirements()).To(BeTrue())
			Expect(waitScheduler.EvaluateInternalInst()).To(BeFalse())
			Expect(cu.advanceVMemLoadRetirements()).To(BeTrue())
			Expect(waitScheduler.EvaluateInternalInst()).To(BeTrue())
		})

		It("services one FIFO head per wave without same-tick carry", func() {
			cu.vmemWideLoadReturnLaneDwordsPerCycle = 4
			otherWf := wavefront.NewWavefront(kernels.NewWavefront())
			otherInst := wavefront.NewInst(insts.NewInst())
			otherInst.FormatType = insts.FLAT
			secondInst := wavefront.NewInst(insts.NewInst())
			secondInst.FormatType = insts.FLAT
			wf.OutstandingVectorMemAccess = 2
			wf.OutstandingScalarMemAccess = 2
			wf.InFlightInsts = 2
			otherWf.OutstandingVectorMemAccess = 1
			otherWf.OutstandingScalarMemAccess = 1
			otherWf.InFlightInsts = 1
			cu.vmemWideReturnAssemblies = map[uint64]*vmemWideReturnAssembly{
				inst.ID: {
					laneDwords: 5,
					activeLanes: map[int]struct{}{
						0: {},
					},
				},
				secondInst.ID: {
					laneDwords: 5,
					activeLanes: map[int]struct{}{
						0: {},
					},
				},
				otherInst.ID: {
					laneDwords: 9,
					activeLanes: map[int]struct{}{
						0: {},
					},
				},
			}
			cu.finishVectorMemLoadReturn(wf, inst)
			cu.finishVectorMemLoadReturn(wf, secondInst)
			cu.finishVectorMemLoadReturn(otherWf, otherInst)
			Expect(cu.pendingVMemRetirements).To(HaveLen(3))

			cu.advanceVMemLoadRetirements()

			Expect(cu.pendingVMemRetirements).To(HaveLen(2))
			Expect(cu.pendingVMemRetirements[0].inst).To(BeIdenticalTo(secondInst))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(1))
			Expect(cu.pendingVMemRetirements[1].remainingCycles).To(Equal(1))
			Expect(wf.OutstandingVectorMemAccess).To(Equal(1))
			Expect(otherWf.OutstandingVectorMemAccess).To(Equal(1))

			cu.advanceVMemLoadRetirements()

			Expect(cu.pendingVMemRetirements).To(BeEmpty())
			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(otherWf.OutstandingVectorMemAccess).To(Equal(0))
		})

		It("starts a final-sibling CU-wide return on the next tick", func() {
			cu.vmemCUWideReturnUnitsPerCycle = 3
			wf.InFlightInsts = 1
			read.CanWaitForCoalesce = false

			sibling := info
			sibling.Read = &memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
				CanWaitForCoalesce: true,
			}
			for i := range sibling.laneInfo {
				sibling.laneInfo[i].reg = insts.VReg(1)
			}
			cu.InFlightVectorMemAccess = append(
				cu.InFlightVectorMemAccess, sibling)

			cu.Tick()

			assembly := cu.vmemWideReturnAssemblies[inst.ID]
			Expect(assembly.laneDwords).To(Equal(4))
			Expect(assembly.activeLanes).To(HaveLen(4))
			Expect(cu.pendingVMemRetirements).To(BeEmpty())

			toVectorMem.incoming = append(
				toVectorMem.incoming,
				memprotocol.DataReadyRsp{
					MsgMeta: messaging.MsgMeta{
						ID:    timing.GetIDGenerator().Generate(),
						RspTo: sibling.Read.ID,
					},
					Data: make([]byte, 16),
				},
			)
			cu.Tick()

			Expect(cu.vmemWideReturnAssemblies).NotTo(HaveKey(inst.ID))
			Expect(cu.pendingVMemRetirements).To(HaveLen(1))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(4))
			Expect(wf.OutstandingVectorMemAccess).To(Equal(1))

			waitInst := wavefront.NewInst(insts.NewInst())
			waitInst.Format = insts.FormatTable[insts.SOPP]
			waitInst.Opcode = 12
			waitInst.VMCNT = 0
			waitInst.LKGMCNT = 0
			wf.SetDynamicInst(waitInst)
			wf.State = wavefront.WfRunning
			wf.InFlightInsts++
			waitScheduler := NewScheduler(cu, nil, nil)
			waitScheduler.internalExecuting = []*wavefront.Wavefront{wf}

			cu.Tick()
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(1))
			Expect(waitScheduler.EvaluateInternalInst()).To(BeFalse())

			cu.Tick()
			Expect(cu.pendingVMemRetirements).To(BeEmpty())
			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(0))
			Expect(waitScheduler.EvaluateInternalInst()).To(BeTrue())

			cu.Tick()
			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(0))
		})

		It("defaults to deterministic fair P1 CU-wide service", func() {
			cu.vmemCUWideReturnUnitsPerCycle = 5
			wf.SIMDID = 0
			secondInst := wavefront.NewInst(insts.NewInst())
			secondInst.FormatType = insts.FLAT
			otherInst := wavefront.NewInst(insts.NewInst())
			otherInst.FormatType = insts.FLAT
			otherWf := wavefront.NewWavefront(kernels.NewWavefront())
			otherWf.SIMDID = 1
			wf.OutstandingVectorMemAccess = 2
			wf.OutstandingScalarMemAccess = 2
			wf.InFlightInsts = 2
			otherWf.OutstandingVectorMemAccess = 1
			otherWf.OutstandingScalarMemAccess = 1
			otherWf.InFlightInsts = 1

			makeX2Assembly := func(activeLanes int) *vmemWideReturnAssembly {
				assembly := &vmemWideReturnAssembly{
					laneDwords:  2 * activeLanes,
					activeLanes: make(map[int]struct{}, activeLanes),
				}
				for lane := 0; lane < activeLanes; lane++ {
					assembly.activeLanes[lane] = struct{}{}
				}
				return assembly
			}
			cu.vmemWideReturnAssemblies = map[uint64]*vmemWideReturnAssembly{
				inst.ID:       makeX2Assembly(7),
				secondInst.ID: makeX2Assembly(3),
				otherInst.ID:  makeX2Assembly(4),
			}
			cu.finishVectorMemLoadReturn(wf, inst)
			cu.finishVectorMemLoadReturn(wf, secondInst)
			cu.finishVectorMemLoadReturn(otherWf, otherInst)

			Expect(cu.pendingVMemRetirements).To(HaveLen(3))
			Expect(cu.pendingVMemRetirements[0].inst).To(BeIdenticalTo(inst))
			Expect(cu.pendingVMemRetirements[1].inst).To(BeIdenticalTo(secondInst))
			Expect(cu.pendingVMemRetirements[2].inst).To(BeIdenticalTo(otherInst))

			cu.advanceVMemLoadRetirements()

			Expect(cu.pendingVMemRetirements).To(HaveLen(3))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(2))
			Expect(cu.pendingVMemRetirements[1].remainingCycles).To(Equal(3))
			Expect(cu.pendingVMemRetirements[2].remainingCycles).To(Equal(4))

			cu.advanceVMemLoadRetirements()

			Expect(cu.pendingVMemRetirements).To(HaveLen(2))
			Expect(cu.pendingVMemRetirements[0].inst).To(BeIdenticalTo(inst))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(2))
			Expect(cu.pendingVMemRetirements[1].inst).To(BeIdenticalTo(secondInst))
			Expect(cu.pendingVMemRetirements[1].remainingCycles).To(Equal(3))
			Expect(wf.OutstandingVectorMemAccess).To(Equal(2))
			Expect(otherWf.OutstandingVectorMemAccess).To(Equal(0))

			cu.advanceVMemLoadRetirements()

			Expect(cu.pendingVMemRetirements).To(BeEmpty())
			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
		})

		It("makes the compatibility P0 setting exactly match explicit P1", func() {
			cu.vmemCUWideReturnUnitsPerCycle = 2
			_, first := makePendingVMemWave(0, 5)
			_, second := makePendingVMemWave(0, 5)
			original := []pendingVMemLoadRetirement{first[0], second[0]}

			cu.pendingVMemRetirements = append(
				[]pendingVMemLoadRetirement(nil), original...)
			cu.advanceVMemLoadRetirements()
			defaultRemaining := []int{
				cu.pendingVMemRetirements[0].remainingCycles,
				cu.pendingVMemRetirements[1].remainingCycles,
			}
			defaultNext := cu.vmemCUWideNextWave

			cu.pendingVMemRetirements = append(
				[]pendingVMemLoadRetirement(nil), original...)
			cu.vmemCUWideWaveOrder = nil
			cu.vmemCUWideNextWave = 0
			cu.vmemCUWideReturnConcurrentWaves = 1
			cu.advanceVMemLoadRetirements()

			Expect([]int{
				cu.pendingVMemRetirements[0].remainingCycles,
				cu.pendingVMemRetirements[1].remainingCycles,
			}).To(Equal(defaultRemaining))
			Expect(cu.vmemCUWideNextWave).To(Equal(defaultNext))
		})

		It("serializes one wave under P2 and carries its same-tick remainder", func() {
			cu.vmemCUWideReturnUnitsPerCycle = 4
			cu.vmemCUWideReturnConcurrentWaves = 2
			wf, retirements := makePendingVMemWave(0, 2, 5)
			_, other := makePendingVMemWave(1, 8)
			cu.pendingVMemRetirements = []pendingVMemLoadRetirement{
				retirements[0], other[0], retirements[1],
			}

			cu.advanceVMemLoadRetirements()

			Expect(cu.pendingVMemRetirements).To(HaveLen(2))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(4))
			Expect(cu.pendingVMemRetirements[1].wf).To(BeIdenticalTo(wf))
			Expect(cu.pendingVMemRetirements[1].remainingCycles).To(Equal(3))
			Expect(wf.OutstandingVectorMemAccess).To(Equal(1))
		})

		It("services two wave FIFOs concurrently under P2", func() {
			cu.vmemCUWideReturnUnitsPerCycle = 2
			cu.vmemCUWideReturnConcurrentWaves = 2
			_, first := makePendingVMemWave(0, 5)
			_, second := makePendingVMemWave(0, 5)
			cu.pendingVMemRetirements = []pendingVMemLoadRetirement{
				first[0], second[0],
			}

			cu.advanceVMemLoadRetirements()

			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(3))
			Expect(cu.pendingVMemRetirements[1].remainingCycles).To(Equal(3))
		})

		It("round-robins a third wave fairly with two service slots", func() {
			cu.vmemCUWideReturnUnitsPerCycle = 1
			cu.vmemCUWideReturnConcurrentWaves = 2
			_, first := makePendingVMemWave(0, 10)
			_, second := makePendingVMemWave(0, 10)
			_, third := makePendingVMemWave(0, 10)
			cu.pendingVMemRetirements = []pendingVMemLoadRetirement{
				first[0], second[0], third[0],
			}

			cu.advanceVMemLoadRetirements()
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(9))
			Expect(cu.pendingVMemRetirements[1].remainingCycles).To(Equal(9))
			Expect(cu.pendingVMemRetirements[2].remainingCycles).To(Equal(10))

			cu.advanceVMemLoadRetirements()
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(8))
			Expect(cu.pendingVMemRetirements[1].remainingCycles).To(Equal(9))
			Expect(cu.pendingVMemRetirements[2].remainingCycles).To(Equal(9))
		})

		It("does not transfer unused budget when P covers every queue", func() {
			cu.vmemCUWideReturnUnitsPerCycle = 2
			cu.vmemCUWideReturnConcurrentWaves = 4
			firstWf, first := makePendingVMemWave(0, 1)
			_, second := makePendingVMemWave(0, 5)
			cu.pendingVMemRetirements = []pendingVMemLoadRetirement{
				first[0], second[0],
			}

			cu.advanceVMemLoadRetirements()

			Expect(firstWf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(cu.pendingVMemRetirements).To(HaveLen(1))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(3))
		})

		It("services waves on different SIMDs with independent budgets", func() {
			cu.vmemCUWideReturnUnitsPerCycle = 2
			cu.vmemCUWideReturnConcurrentWaves = 2
			_, first := makePendingVMemWave(0, 4)
			_, second := makePendingVMemWave(3, 4)
			cu.pendingVMemRetirements = []pendingVMemLoadRetirement{
				first[0], second[0],
			}

			cu.advanceVMemLoadRetirements()

			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(2))
			Expect(cu.pendingVMemRetirements[1].remainingCycles).To(Equal(2))
		})
	})

	Context("handle write done respond from ToVectorMem port", func() {
		var (
			rawWf    *kernels.Wavefront
			inst     *wavefront.Inst
			wf       *wavefront.Wavefront
			info     VectorMemAccessInfo
			writeReq *memprotocol.WriteReq
		)

		BeforeEach(func() {
			rawWf = grid.WorkGroups[0].Wavefronts[0]
			inst = wavefront.NewInst(insts.NewInst())
			inst.FormatType = insts.FLAT
			wf = wavefront.NewWavefront(rawWf)
			wf.SIMDID = 0
			wf.SetDynamicInst(inst)
			wf.VRegOffset = 0
			wf.OutstandingVectorMemAccess = 1
			wf.OutstandingScalarMemAccess = 1

			writeReq = &memprotocol.WriteReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
				Address:            0x100,
				CanWaitForCoalesce: true,
			}

			info = VectorMemAccessInfo{}
			info.Wavefront = wf
			info.Inst = inst
			info.Write = writeReq
			cu.InFlightVectorMemAccess = append(
				cu.InFlightVectorMemAccess, info)

			doneRsp := memprotocol.WriteDoneRsp{
				MsgMeta: messaging.MsgMeta{
					ID:    timing.GetIDGenerator().Generate(),
					RspTo: writeReq.ID,
				},
			}
			toVectorMem.incoming = append(toVectorMem.incoming, doneRsp)
		})

		It("should handle vector data store return and the return is not "+
			"the last one from an instruction", func() {
			// The request generated last can return before an older sibling.
			writeReq.CanWaitForCoalesce = false
			sibling := info
			sibling.Write = &memprotocol.WriteReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
			}
			cu.InFlightVectorMemAccess = append(
				cu.InFlightVectorMemAccess, sibling)

			madeProgress := cu.processInputFromVectorMem()

			Expect(wf.OutstandingVectorMemAccess).To(Equal(1))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(1))
			Expect(cu.InFlightVectorMemAccess).To(HaveLen(1))
			Expect(madeProgress).To(BeTrue())
		})

		It("should handle vector data store return and the return is the "+
			"last one from an instruction", func() {
			cu.vmemReturnFanoutLaneDwordsPerCycle = 1
			writeReq.CanWaitForCoalesce = false

			cu.processInputFromVectorMem()

			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(0))
			Expect(cu.InFlightVectorMemAccess).To(HaveLen(0))
		})

		It("does not route stores through the CU-wide return FIFO", func() {
			cu.vmemCUWideReturnUnitsPerCycle = 1
			writeReq.CanWaitForCoalesce = false

			cu.processInputFromVectorMem()

			Expect(wf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(wf.OutstandingScalarMemAccess).To(Equal(0))
			Expect(cu.pendingVMemRetirements).To(BeEmpty())
		})
	})

	Context("should handle flush request", func() {
		It("should handle a pipeline flush request from CU", func() {
			req := protocol.CUPipelineFlushReq{
				MsgMeta: messaging.MsgMeta{
					ID:  timing.GetIDGenerator().Generate(),
					Src: "CP",
					Dst: cu.ToCP.AsRemote(),
				},
			}

			toCP.incoming = append(toCP.incoming, req)

			cu.processInputFromCP()

			Expect(cu.comp.State.IsFlushing).To(BeTrue())
			Expect(cu.comp.State.HasFlushReq).To(BeTrue())
			Expect(cu.comp.State.FlushReqID).To(Equal(req.ID))
			Expect(cu.comp.State.FlushReqSrc).To(Equal(req.Src))
			Expect(toCP.incoming).To(HaveLen(0))
		})

		It("should flush internal CU buffers", func() {
			info := new(InstFetchReqInfo)
			cu.InFlightInstFetch = append(cu.InFlightInstFetch, info)

			scalarMemInfo := new(ScalarMemAccessInfo)
			cu.InFlightScalarMemAccess = append(
				cu.InFlightScalarMemAccess, scalarMemInfo)

			vectorMemInfo := VectorMemAccessInfo{}
			cu.InFlightVectorMemAccess = append(
				cu.InFlightVectorMemAccess, vectorMemInfo)

			cu.flushCUBuffers()

			Expect(cu.InFlightInstFetch).To(BeNil())
			Expect(cu.InFlightVectorMemAccess).To(BeNil())
			Expect(cu.InFlightScalarMemAccess).To(BeNil())
		})

		It("preserves total-return assembly through flush", func() {
			cu.vmemLoadReturnLaneDwordsPerCycle = 4
			assemblyInst := wavefront.NewInst(insts.NewInst())
			assemblyInst.FormatType = insts.FLAT
			assemblyReq := &memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
			}
			cu.InFlightVectorMemAccess = []VectorMemAccessInfo{
				{
					Read:      assemblyReq,
					Wavefront: wavefront.NewWavefront(kernels.NewWavefront()),
					Inst:      assemblyInst,
				},
			}
			cu.vmemReturnAssemblies = map[uint64]int{assemblyInst.ID: 7}
			cu.comp.State.HasFlushReq = true
			cu.comp.State.FlushReqID = timing.GetIDGenerator().Generate()
			cu.comp.State.FlushReqSrc = "CP"

			Expect(cu.flushPipeline()).To(BeTrue())

			Expect(cu.vmemReturnAssemblies[assemblyInst.ID]).To(Equal(7))
			Expect(cu.shadowInFlightVectorMemAccess).To(HaveLen(1))
			Expect(cu.comp.State.IsPaused).To(BeTrue())
			Expect(cu.runPipeline()).To(BeFalse())
			Expect(cu.vmemReturnAssemblies[assemblyInst.ID]).To(Equal(7))
		})

		It("preserves return state through flush and restart", func() {
			cu.vmemWideLoadReturnLaneDwordsPerCycle = 4
			assemblyInst := wavefront.NewInst(insts.NewInst())
			assemblyInst.FormatType = insts.FLAT
			assemblyWf := wavefront.NewWavefront(kernels.NewWavefront())
			assemblyReq := &memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
			}
			cu.InFlightVectorMemAccess = []VectorMemAccessInfo{
				{
					Read:      assemblyReq,
					Wavefront: assemblyWf,
					Inst:      assemblyInst,
				},
			}
			cu.vmemWideReturnAssemblies = map[uint64]*vmemWideReturnAssembly{
				assemblyInst.ID: {
					laneDwords: 7,
					activeLanes: map[int]struct{}{
						0: {},
						1: {},
					},
				},
			}

			pendingInst := wavefront.NewInst(insts.NewInst())
			pendingInst.FormatType = insts.FLAT
			pendingWf := wavefront.NewWavefront(kernels.NewWavefront())
			pendingWf.OutstandingVectorMemAccess = 1
			pendingWf.OutstandingScalarMemAccess = 1
			pendingWf.InFlightInsts = 1
			cu.pendingVMemRetirements = []pendingVMemLoadRetirement{
				{wf: pendingWf, inst: pendingInst, remainingCycles: 3},
			}
			cu.comp.State.HasFlushReq = true
			cu.comp.State.FlushReqID = timing.GetIDGenerator().Generate()
			cu.comp.State.FlushReqSrc = "CP"

			Expect(cu.flushPipeline()).To(BeTrue())

			Expect(cu.vmemWideReturnAssemblies[assemblyInst.ID].laneDwords).
				To(Equal(7))
			Expect(cu.vmemWideReturnAssemblies[assemblyInst.ID].activeLanes).
				To(HaveLen(2))
			Expect(cu.pendingVMemRetirements).To(HaveLen(1))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(3))
			Expect(cu.shadowInFlightVectorMemAccess).To(HaveLen(1))
			Expect(cu.comp.State.IsPaused).To(BeTrue())

			Expect(cu.runPipeline()).To(BeFalse())
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(3))
			Expect(cu.sendToCP()).To(BeTrue())
			restartReq := protocol.CUPipelineRestartReq{
				MsgMeta: messaging.MsgMeta{
					ID:  timing.GetIDGenerator().Generate(),
					Src: "CP",
					Dst: cu.ToCP.AsRemote(),
				},
			}
			toCP.incoming = append(toCP.incoming, restartReq)
			Expect(cu.processInputFromCP()).To(BeTrue())
			Expect(cu.comp.State.IsSendingOutShadowBufferReqs).To(BeTrue())
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(3))

			Expect(cu.checkShadowBuffers()).To(BeTrue())
			Expect(cu.checkShadowBuffers()).To(BeTrue())
			Expect(cu.comp.State.IsPaused).To(BeFalse())
			Expect(cu.vmemWideReturnAssemblies[assemblyInst.ID].laneDwords).
				To(Equal(7))

			for remaining := 2; remaining > 0; remaining-- {
				Expect(cu.advanceVMemLoadRetirements()).To(BeTrue())
				Expect(cu.pendingVMemRetirements[0].remainingCycles).
					To(Equal(remaining))
				Expect(pendingWf.OutstandingVectorMemAccess).To(Equal(1))
			}
			Expect(cu.advanceVMemLoadRetirements()).To(BeTrue())
			Expect(cu.pendingVMemRetirements).To(BeEmpty())
			Expect(pendingWf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(pendingWf.OutstandingScalarMemAccess).To(Equal(0))
			Expect(pendingWf.InFlightInsts).To(Equal(0))
			Expect(cu.advanceVMemLoadRetirements()).To(BeFalse())
			Expect(pendingWf.OutstandingVectorMemAccess).To(Equal(0))
		})

		It("preserves CU-wide wave queues and round-robin state through flush and restart", func() {
			cu.vmemCUWideReturnUnitsPerCycle = 4
			cu.vmemCUWideReturnConcurrentWaves = 1
			assemblyInst := wavefront.NewInst(insts.NewInst())
			assemblyInst.FormatType = insts.FLAT
			assemblyWf := wavefront.NewWavefront(kernels.NewWavefront())
			assemblyReq := &memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID: timing.GetIDGenerator().Generate(),
				},
			}
			cu.InFlightVectorMemAccess = []VectorMemAccessInfo{
				{
					Read:      assemblyReq,
					Wavefront: assemblyWf,
					Inst:      assemblyInst,
				},
			}
			cu.vmemWideReturnAssemblies = map[uint64]*vmemWideReturnAssembly{
				assemblyInst.ID: {
					laneDwords: 6,
					activeLanes: map[int]struct{}{
						0: {},
						1: {},
						2: {},
					},
				},
			}

			firstInst := wavefront.NewInst(insts.NewInst())
			firstInst.FormatType = insts.FLAT
			firstWf := wavefront.NewWavefront(kernels.NewWavefront())
			firstWf.OutstandingVectorMemAccess = 1
			firstWf.OutstandingScalarMemAccess = 1
			firstWf.InFlightInsts = 1
			secondInst := wavefront.NewInst(insts.NewInst())
			secondInst.FormatType = insts.FLAT
			secondWf := wavefront.NewWavefront(kernels.NewWavefront())
			secondWf.SIMDID = 1
			secondWf.OutstandingVectorMemAccess = 1
			secondWf.OutstandingScalarMemAccess = 1
			secondWf.InFlightInsts = 1
			cu.pendingVMemRetirements = []pendingVMemLoadRetirement{
				{wf: firstWf, inst: firstInst, remainingCycles: 6},
				{wf: secondWf, inst: secondInst, remainingCycles: 3},
			}
			cu.vmemCUWideWaveOrder = []*wavefront.Wavefront{firstWf, secondWf}
			cu.vmemCUWideNextWave = 1
			cu.comp.State.HasFlushReq = true
			cu.comp.State.FlushReqID = timing.GetIDGenerator().Generate()
			cu.comp.State.FlushReqSrc = "CP"

			Expect(cu.flushPipeline()).To(BeTrue())
			Expect(cu.comp.State.IsPaused).To(BeTrue())
			Expect(cu.pendingVMemRetirements).To(HaveLen(2))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(6))
			Expect(cu.pendingVMemRetirements[1].remainingCycles).To(Equal(3))
			Expect(cu.vmemCUWideWaveOrder).To(Equal(
				[]*wavefront.Wavefront{firstWf, secondWf},
			))
			Expect(cu.vmemCUWideNextWave).To(Equal(1))
			Expect(cu.vmemWideReturnAssemblies[assemblyInst.ID].laneDwords).
				To(Equal(6))

			Expect(cu.runPipeline()).To(BeFalse())
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(6))
			Expect(cu.sendToCP()).To(BeTrue())
			restartReq := protocol.CUPipelineRestartReq{
				MsgMeta: messaging.MsgMeta{
					ID:  timing.GetIDGenerator().Generate(),
					Src: "CP",
					Dst: cu.ToCP.AsRemote(),
				},
			}
			toCP.incoming = append(toCP.incoming, restartReq)
			Expect(cu.processInputFromCP()).To(BeTrue())
			Expect(cu.comp.State.IsSendingOutShadowBufferReqs).To(BeTrue())
			Expect(cu.checkShadowBuffers()).To(BeTrue())
			Expect(cu.checkShadowBuffers()).To(BeTrue())
			Expect(cu.comp.State.IsPaused).To(BeFalse())
			Expect(cu.vmemWideReturnAssemblies[assemblyInst.ID].laneDwords).
				To(Equal(6))
			Expect(cu.vmemCUWideNextWave).To(Equal(1))

			Expect(cu.advanceVMemLoadRetirements()).To(BeTrue())
			Expect(cu.pendingVMemRetirements).To(HaveLen(1))
			Expect(cu.pendingVMemRetirements[0].inst).To(BeIdenticalTo(firstInst))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(6))
			Expect(firstWf.OutstandingVectorMemAccess).To(Equal(1))
			Expect(secondWf.OutstandingVectorMemAccess).To(Equal(0))

			Expect(cu.advanceVMemLoadRetirements()).To(BeTrue())
			Expect(cu.pendingVMemRetirements).To(HaveLen(1))
			Expect(cu.pendingVMemRetirements[0].inst).To(BeIdenticalTo(firstInst))
			Expect(cu.pendingVMemRetirements[0].remainingCycles).To(Equal(2))
			Expect(firstWf.OutstandingVectorMemAccess).To(Equal(1))

			Expect(cu.advanceVMemLoadRetirements()).To(BeTrue())
			Expect(cu.pendingVMemRetirements).To(BeEmpty())
			Expect(firstWf.OutstandingVectorMemAccess).To(Equal(0))
			Expect(cu.advanceVMemLoadRetirements()).To(BeFalse())
		})

		It("should handle a restart request", func() {
			cu.comp.State.IsPaused = true

			req := protocol.CUPipelineRestartReq{
				MsgMeta: messaging.MsgMeta{
					ID:  timing.GetIDGenerator().Generate(),
					Src: "CP",
					Dst: cu.ToCP.AsRemote(),
				},
			}

			toCP.incoming = append(toCP.incoming, req)

			cu.processInputFromCP()

			Expect(toCP.sent).To(HaveLen(1))
			Expect(toCP.sent[0]).To(
				BeAssignableToTypeOf(protocol.CUPipelineRestartRsp{}))
			Expect(cu.comp.State.IsPaused).To(BeTrue())
			Expect(cu.comp.State.IsSendingOutShadowBufferReqs).To(BeTrue())
		})

		It("should flush the full CU", func() {
			req := protocol.CUPipelineFlushReq{
				MsgMeta: messaging.MsgMeta{
					ID:  timing.GetIDGenerator().Generate(),
					Src: "CP",
					Dst: cu.ToCP.AsRemote(),
				},
			}

			cu.comp.State.HasFlushReq = true
			cu.comp.State.FlushReqID = req.ID
			cu.comp.State.FlushReqSrc = req.Src

			info := new(InstFetchReqInfo)
			cu.InFlightInstFetch = append(cu.InFlightInstFetch, info)

			scalarMemInfo := new(ScalarMemAccessInfo)
			cu.InFlightScalarMemAccess = append(
				cu.InFlightScalarMemAccess, scalarMemInfo)

			vectorMemInfo := VectorMemAccessInfo{}
			cu.InFlightVectorMemAccess = append(
				cu.InFlightVectorMemAccess, vectorMemInfo)

			cu.flushPipeline()

			Expect(cu.InFlightInstFetch).To(BeNil())
			Expect(cu.InFlightVectorMemAccess).To(BeNil())
			Expect(cu.InFlightScalarMemAccess).To(BeNil())

			Expect(cu.shadowInFlightInstFetch).To(Not(BeNil()))
			Expect(cu.shadowInFlightVectorMemAccess).To(Not(BeNil()))
			Expect(cu.shadowInFlightScalarMemAccess).To(Not(BeNil()))

			Expect(branchUnit.flushed).To(BeTrue())
			Expect(scalarUnit.flushed).To(BeTrue())
			Expect(scalarDecoder.flushed).To(BeTrue())
			Expect(simdUnit.flushed).To(BeTrue())
			Expect(vectorDecoder.flushed).To(BeTrue())
			Expect(ldsUnit.flushed).To(BeTrue())
			Expect(ldsDecoder.flushed).To(BeTrue())
			Expect(vectorMemDecoder.flushed).To(BeTrue())
			Expect(vectorMemUnit.flushed).To(BeTrue())

			Expect(cu.comp.State.HasPendingCPRsp).To(BeTrue())
			Expect(cu.comp.State.IsFlushing).To(BeFalse())
			Expect(cu.comp.State.IsPaused).To(BeTrue())
		})

		It("should not restart a CU where there are shadow buffer reqs "+
			"pending", func() {
			req := memprotocol.ReadReq{
				MsgMeta: messaging.MsgMeta{
					ID:  timing.GetIDGenerator().Generate(),
					Src: cu.ToInstMem.AsRemote(),
					Dst: cu.comp.State.InstMem,
				},
				Address:        0x100,
				AccessByteSize: 64,
			}

			info := new(InstFetchReqInfo)
			info.Req = req
			cu.shadowInFlightInstFetch = append(
				cu.shadowInFlightInstFetch, info)

			// Mem accesses always carry their instruction in production (set
			// in executeSMEMLoad / the coalescer); the shadow resend parents
			// the replacement req_out task on it.
			inst := wavefront.NewInst(insts.NewInst())

			scalarMemInfo := new(ScalarMemAccessInfo)
			scalarMemInfo.Req = req
			scalarMemInfo.Inst = inst
			cu.shadowInFlightScalarMemAccess = append(
				cu.shadowInFlightScalarMemAccess, scalarMemInfo)

			vectorMemInfo := VectorMemAccessInfo{}
			readCopy := req
			vectorMemInfo.Read = &readCopy
			vectorMemInfo.Inst = inst
			cu.shadowInFlightVectorMemAccess = append(
				cu.shadowInFlightVectorMemAccess, vectorMemInfo)

			cu.checkShadowBuffers()

			Expect(toInstMem.sent).To(HaveLen(1))
			Expect(toScalarMem.sent).To(HaveLen(1))
			Expect(toVectorMem.sent).To(HaveLen(1))
		})

		It("should restart a CU where there are no shadow buffer reqs "+
			"pending", func() {
			cu.shadowInFlightInstFetch = nil
			cu.shadowInFlightScalarMemAccess = nil
			cu.shadowInFlightVectorMemAccess = nil

			cu.checkShadowBuffers()

			Expect(cu.comp.State.IsPaused).To(BeFalse())
		})
	})
})
