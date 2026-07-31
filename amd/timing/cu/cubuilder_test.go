package cu

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sarchlab/akita/v5/messaging"
	"github.com/sarchlab/akita/v5/modeling"
	"github.com/sarchlab/akita/v5/timing"
)

var _ = Describe("Builder", func() {
	It("should build a fully equipped compute unit", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)

		comp := MakeBuilder().
			WithRegistrar(reg).
			WithSpec(DefaultSpec()).
			Build("GPU.CU")

		cuMW := MiddlewareOf(comp)

		Expect(cuMW.Scheduler).NotTo(BeNil())
		Expect(cuMW.BranchUnit).NotTo(BeNil())
		Expect(cuMW.ScalarUnit).NotTo(BeNil())
		Expect(cuMW.ScalarDecoder).NotTo(BeNil())
		Expect(cuMW.VectorDecoder).NotTo(BeNil())
		Expect(cuMW.LDSDecoder).NotTo(BeNil())
		Expect(cuMW.LDSUnit).NotTo(BeNil())
		Expect(cuMW.VectorMemDecoder).NotTo(BeNil())
		Expect(cuMW.VectorMemUnit).NotTo(BeNil())
		Expect(cuMW.SIMDUnit).To(HaveLen(4))
		Expect(cuMW.VRegFile).To(HaveLen(4))
		Expect(cuMW.SRegFile).NotTo(BeNil())
		Expect(cuMW.WfPools).To(HaveLen(4))
		Expect(cuMW.WfDispatcher).NotTo(BeNil())
		Expect(cuMW.Decoder).NotTo(BeNil())

		Expect(comp.Resources().Decoder).NotTo(BeNil())
		Expect(comp.Resources().ALU).NotTo(BeNil())

		// All five ports must be declared so external code can assign them.
		for _, portName := range []string{
			DispatchPortName, CtrlPortName,
			InstMemPortName, ScalarMemPortName, VectorMemPortName,
		} {
			p := messaging.NewPort(comp, 4, 4, "GPU.CU."+portName)
			Expect(func() { comp.AssignPort(portName, p) }).NotTo(Panic())
		}

		view := DispatcherView{CU: comp}
		Expect(view.WfPoolSizes()).To(Equal([]int{10, 10, 10, 10}))
		Expect(view.VRegCounts()).To(
			Equal([]int{16384, 16384, 16384, 16384}))
		Expect(view.SRegCount()).To(Equal(3200))
		Expect(view.LDSBytes()).To(Equal(64 * 1024))
		Expect(view.DispatchingPort()).To(
			Equal(messaging.RemotePort("GPU.CU.Top")))
		Expect(view.ControlPort()).To(
			Equal(messaging.RemotePort("GPU.CU.Ctrl")))
	})

	It("rejects an invalid dependent-load timing configuration", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)
		spec := DefaultSpec()
		spec.DependentLoadIssuePenalty = 1
		spec.DependentLoadMaxAge = 0

		Expect(func() {
			MakeBuilder().
				WithRegistrar(reg).
				WithSpec(spec).
				Build("GPU.CU")
		}).To(PanicWith(
			"cu: DependentLoadMaxAge must be positive when dependency tracking is enabled",
		))
	})

	It("rejects an inverted dependent-load age window", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)
		spec := DefaultSpec()
		spec.DependentLoadIssuePenalty = 1
		spec.DependentLoadMinAge = 4
		spec.DependentLoadMaxAge = 3

		Expect(func() {
			MakeBuilder().
				WithRegistrar(reg).
				WithSpec(spec).
				Build("GPU.CU")
		}).To(PanicWith("cu: dependent-load age window is invalid"))
	})

	It("rejects negative load-width filters", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)
		spec := DefaultSpec()
		spec.SplitLineLoadMaxDwords = -1

		Expect(func() {
			MakeBuilder().
				WithRegistrar(reg).
				WithSpec(spec).
				Build("GPU.CU")
		}).To(PanicWith("cu: SplitLineLoadMaxDwords cannot be negative"))
	})

	It("rejects an incomplete wide-write far tier", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)
		spec := DefaultSpec()
		spec.MaxWideWriteStridePenalty = 20
		spec.MaxWideWriteStrideFarPenalty = 30
		spec.MaxWideWriteStrideFarMinDistanceLines = 1

		Expect(func() {
			MakeBuilder().
				WithRegistrar(reg).
				WithSpec(spec).
				Build("GPU.CU")
		}).To(PanicWith(
			"cu: wide-write far penalty requires a distance of at least 2 lines",
		))
	})

	It("rejects a wide-write far tier without a near tier", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)
		spec := DefaultSpec()
		spec.MaxWideWriteStrideFarPenalty = 30
		spec.MaxWideWriteStrideFarMinDistanceLines = 4

		Expect(func() {
			MakeBuilder().
				WithRegistrar(reg).
				WithSpec(spec).
				Build("GPU.CU")
		}).To(PanicWith(
			"cu: wide-write far penalty requires a near penalty",
		))
	})
})
