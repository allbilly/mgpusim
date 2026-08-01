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
		Expect(comp.Spec().VMemCUWideReturnUnitsPerCycle).To(Equal(0))

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

	It("propagates the vector-memory return fanout bandwidth", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)

		comp := MakeBuilder().
			WithRegistrar(reg).
			WithVMemReturnFanoutLaneDwordsPerCycle(7).
			Build("GPU.CU")

		Expect(comp.Spec().VMemReturnFanoutLaneDwordsPerCycle).To(Equal(7))
		Expect(MiddlewareOf(comp).vmemReturnFanoutLaneDwordsPerCycle).To(Equal(7))
	})

	It("rejects a negative vector-memory return fanout bandwidth", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)

		Expect(func() {
			MakeBuilder().
				WithRegistrar(reg).
				WithVMemReturnFanoutLaneDwordsPerCycle(-1).
				Build("GPU.CU")
		}).To(PanicWith(
			"cu: VMemReturnFanoutLaneDwordsPerCycle cannot be negative",
		))
	})

	It("propagates the vector-memory load return bandwidth", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)

		comp := MakeBuilder().
			WithRegistrar(reg).
			WithVMemLoadReturnLaneDwordsPerCycle(9).
			Build("GPU.CU")

		Expect(comp.Spec().VMemLoadReturnLaneDwordsPerCycle).To(Equal(9))
		Expect(MiddlewareOf(comp).vmemLoadReturnLaneDwordsPerCycle).To(Equal(9))
	})

	It("rejects a negative vector-memory load return bandwidth", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)

		Expect(func() {
			MakeBuilder().
				WithRegistrar(reg).
				WithVMemLoadReturnLaneDwordsPerCycle(-1).
				Build("GPU.CU")
		}).To(PanicWith(
			"cu: VMemLoadReturnLaneDwordsPerCycle cannot be negative",
		))
	})

	It("propagates the wide vector-memory load return bandwidth", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)

		comp := MakeBuilder().
			WithRegistrar(reg).
			WithVMemWideLoadReturnLaneDwordsPerCycle(11).
			Build("GPU.CU")

		Expect(comp.Spec().VMemWideLoadReturnLaneDwordsPerCycle).To(Equal(11))
		Expect(MiddlewareOf(comp).vmemWideLoadReturnLaneDwordsPerCycle).
			To(Equal(11))
	})

	It("rejects a negative wide vector-memory load return bandwidth", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)

		Expect(func() {
			MakeBuilder().
				WithRegistrar(reg).
				WithVMemWideLoadReturnLaneDwordsPerCycle(-1).
				Build("GPU.CU")
		}).To(PanicWith(
			"cu: VMemWideLoadReturnLaneDwordsPerCycle cannot be negative",
		))
	})

	It("propagates the CU-wide vector-memory return budget", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)

		comp := MakeBuilder().
			WithRegistrar(reg).
			WithVMemCUWideReturnUnitsPerCycle(13).
			Build("GPU.CU")

		Expect(comp.Spec().VMemCUWideReturnUnitsPerCycle).To(Equal(13))
		Expect(MiddlewareOf(comp).vmemCUWideReturnUnitsPerCycle).To(Equal(13))
	})

	It("rejects a negative CU-wide vector-memory return budget", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)

		Expect(func() {
			MakeBuilder().
				WithRegistrar(reg).
				WithVMemCUWideReturnUnitsPerCycle(-1).
				Build("GPU.CU")
		}).To(PanicWith(
			"cu: VMemCUWideReturnUnitsPerCycle cannot be negative",
		))
	})

	It("rejects simultaneous vector-memory return bandwidth models", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)

		Expect(func() {
			MakeBuilder().
				WithRegistrar(reg).
				WithVMemReturnFanoutLaneDwordsPerCycle(7).
				WithVMemLoadReturnLaneDwordsPerCycle(9).
				Build("GPU.CU")
		}).To(PanicWith(
			"cu: vector-memory return bandwidth models are mutually exclusive",
		))
	})

	It("rejects the wide model with either existing return model", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)

		for _, configure := range []func(Builder) Builder{
			func(b Builder) Builder {
				return b.WithVMemReturnFanoutLaneDwordsPerCycle(7)
			},
			func(b Builder) Builder {
				return b.WithVMemLoadReturnLaneDwordsPerCycle(9)
			},
		} {
			Expect(func() {
				configure(MakeBuilder()).
					WithRegistrar(reg).
					WithVMemWideLoadReturnLaneDwordsPerCycle(11).
					Build("GPU.CU")
			}).To(PanicWith(
				"cu: vector-memory return bandwidth models are mutually exclusive",
			))
		}
	})

	It("rejects the CU-wide model with every existing return model", func() {
		engine := timing.NewSerialEngine()
		reg := modeling.NewStandaloneRegistrar(engine)

		for _, configure := range []func(Builder) Builder{
			func(b Builder) Builder {
				return b.WithVMemReturnFanoutLaneDwordsPerCycle(7)
			},
			func(b Builder) Builder {
				return b.WithVMemLoadReturnLaneDwordsPerCycle(9)
			},
			func(b Builder) Builder {
				return b.WithVMemWideLoadReturnLaneDwordsPerCycle(11)
			},
		} {
			Expect(func() {
				configure(MakeBuilder()).
					WithRegistrar(reg).
					WithVMemCUWideReturnUnitsPerCycle(13).
					Build("GPU.CU")
			}).To(PanicWith(
				"cu: vector-memory return bandwidth models are mutually exclusive",
			))
		}
	})
})
