package requestlimitedconnection

import (
	"testing"

	"github.com/sarchlab/akita/v5/mem/memprotocol"
	"github.com/sarchlab/akita/v5/messaging"
	"github.com/sarchlab/akita/v5/modeling"
	"github.com/sarchlab/akita/v5/timing"
)

func TestLimitsRequestsButNotResponses(t *testing.T) {
	engine := timing.NewSerialEngine()
	connection := MakeBuilder().
		WithRegistrar(modeling.NewStandaloneRegistrar(engine)).
		WithSpec(Spec{
			Freq:                   1 * timing.GHz,
			RequestRateNumerator:   2,
			RequestRateDenominator: 1,
		}).
		Build("Connection")
	requester := messaging.NewPort(nil, 8, 8, "Requester")
	memory := messaging.NewPort(nil, 8, 8, "Memory")
	connection.PlugIn(requester)
	connection.PlugIn(memory)

	for range 3 {
		requester.Send(memprotocol.ReadReq{
			MsgMeta: messaging.MsgMeta{
				ID:  timing.GetIDGenerator().Generate(),
				Src: requester.AsRemote(),
				Dst: memory.AsRemote(),
			},
			AccessByteSize: 64,
		})
	}

	connection.Tick()

	if got := memory.NumIncoming(); got != 2 {
		t.Fatalf("delivered %d requests in one cycle, want 2", got)
	}
	if got := requester.NumOutgoing(); got != 1 {
		t.Fatalf("left %d requests queued, want 1", got)
	}

	for range 3 {
		memory.Send(memprotocol.DataReadyRsp{
			MsgMeta: messaging.MsgMeta{
				ID:    timing.GetIDGenerator().Generate(),
				Src:   memory.AsRemote(),
				Dst:   requester.AsRemote(),
				RspTo: 1,
			},
		})
	}

	connection.Tick()

	if got := requester.NumIncoming(); got != 3 {
		t.Fatalf("delivered %d responses in one cycle, want 3", got)
	}
}

func TestZeroRequestLimitIsUnlimited(t *testing.T) {
	engine := timing.NewSerialEngine()
	connection := MakeBuilder().
		WithRegistrar(modeling.NewStandaloneRegistrar(engine)).
		Build("Connection")
	requester := messaging.NewPort(nil, 8, 8, "Requester")
	memory := messaging.NewPort(nil, 8, 8, "Memory")
	connection.PlugIn(requester)
	connection.PlugIn(memory)

	for range 4 {
		requester.Send(memprotocol.WriteReq{
			MsgMeta: messaging.MsgMeta{
				ID:  timing.GetIDGenerator().Generate(),
				Src: requester.AsRemote(),
				Dst: memory.AsRemote(),
			},
			Data: make([]byte, 64),
		})
	}

	connection.Tick()

	if got := memory.NumIncoming(); got != 4 {
		t.Fatalf("delivered %d requests with unlimited spec, want 4", got)
	}
}

func TestWriteFilterLeavesReadsUnlimited(t *testing.T) {
	engine := timing.NewSerialEngine()
	connection := MakeBuilder().
		WithRegistrar(modeling.NewStandaloneRegistrar(engine)).
		WithSpec(Spec{
			Freq:                   1 * timing.GHz,
			RequestRateNumerator:   1,
			RequestRateDenominator: 1,
			BurstRequests:          1,
			RequestFilter:          WriteRequests,
		}).
		Build("Connection")
	requester := messaging.NewPort(nil, 8, 8, "Requester")
	memory := messaging.NewPort(nil, 8, 8, "Memory")
	connection.PlugIn(requester)
	connection.PlugIn(memory)

	requester.Send(memprotocol.ReadReq{
		MsgMeta: messaging.MsgMeta{
			ID:  timing.GetIDGenerator().Generate(),
			Src: requester.AsRemote(),
			Dst: memory.AsRemote(),
		},
		AccessByteSize: 64,
	})
	for range 2 {
		requester.Send(memprotocol.WriteReq{
			MsgMeta: messaging.MsgMeta{
				ID:  timing.GetIDGenerator().Generate(),
				Src: requester.AsRemote(),
				Dst: memory.AsRemote(),
			},
			Data: make([]byte, 64),
		})
	}

	connection.Tick()

	if got := memory.NumIncoming(); got != 2 {
		t.Fatalf("delivered %d messages, want unlimited read plus one write", got)
	}
	if got := requester.NumOutgoing(); got != 1 {
		t.Fatalf("left %d messages queued, want one throttled write", got)
	}
}

func TestAllowsConfiguredInitialBurst(t *testing.T) {
	engine := timing.NewSerialEngine()
	connection := MakeBuilder().
		WithRegistrar(modeling.NewStandaloneRegistrar(engine)).
		WithSpec(Spec{
			Freq:                   1 * timing.GHz,
			RequestRateNumerator:   1,
			RequestRateDenominator: 1,
			BurstRequests:          3,
		}).
		Build("Connection")
	requester := messaging.NewPort(nil, 8, 8, "Requester")
	memory := messaging.NewPort(nil, 8, 8, "Memory")
	connection.PlugIn(requester)
	connection.PlugIn(memory)

	for range 4 {
		requester.Send(memprotocol.ReadReq{
			MsgMeta: messaging.MsgMeta{
				ID:  timing.GetIDGenerator().Generate(),
				Src: requester.AsRemote(),
				Dst: memory.AsRemote(),
			},
			AccessByteSize: 64,
		})
	}

	connection.Tick()

	if got := memory.NumIncoming(); got != 3 {
		t.Fatalf("delivered %d requests from initial burst, want 3", got)
	}
}

func TestSupportsFractionalRequestRate(t *testing.T) {
	engine := timing.NewSerialEngine()
	connection := MakeBuilder().
		WithRegistrar(modeling.NewStandaloneRegistrar(engine)).
		WithSpec(Spec{
			Freq:                   1 * timing.GHz,
			RequestRateNumerator:   3,
			RequestRateDenominator: 2,
		}).
		Build("Connection")
	requester := messaging.NewPort(nil, 8, 8, "Requester")
	memory := messaging.NewPort(nil, 8, 8, "Memory")
	connection.PlugIn(requester)
	connection.PlugIn(memory)

	for range 4 {
		requester.Send(memprotocol.ReadReq{
			MsgMeta: messaging.MsgMeta{
				ID:  timing.GetIDGenerator().Generate(),
				Src: requester.AsRemote(),
				Dst: memory.AsRemote(),
			},
			AccessByteSize: 64,
		})
	}

	connection.Tick()
	connection.Tick()

	if got := memory.NumIncoming(); got != 3 {
		t.Fatalf("delivered %d requests in two cycles, want 3", got)
	}
}

func TestKeepsTickingWhileFractionalCreditAccumulates(t *testing.T) {
	engine := timing.NewSerialEngine()
	connection := MakeBuilder().
		WithRegistrar(modeling.NewStandaloneRegistrar(engine)).
		WithSpec(Spec{
			Freq:                   1 * timing.GHz,
			RequestRateNumerator:   1,
			RequestRateDenominator: 2,
		}).
		Build("Connection")
	requester := messaging.NewPort(nil, 8, 8, "Requester")
	memory := messaging.NewPort(nil, 8, 8, "Memory")
	connection.PlugIn(requester)
	connection.PlugIn(memory)
	requester.Send(memprotocol.ReadReq{
		MsgMeta: messaging.MsgMeta{
			ID:  timing.GetIDGenerator().Generate(),
			Src: requester.AsRemote(),
			Dst: memory.AsRemote(),
		},
		AccessByteSize: 64,
	})

	if madeProgress := connection.Tick(); !madeProgress {
		t.Fatal("connection stopped before enough fractional credit accumulated")
	}
	if got := memory.NumIncoming(); got != 0 {
		t.Fatalf("delivered %d requests with half a credit, want 0", got)
	}

	connection.Tick()
	if got := memory.NumIncoming(); got != 1 {
		t.Fatalf("delivered %d requests after a full credit, want 1", got)
	}
}

func TestRefillAccountsForIdleCycles(t *testing.T) {
	got := refillRequestCredit(1, 16, 2, 5)
	if got != 11 {
		t.Fatalf("refilled to %d credits, want 11", got)
	}

	got = refillRequestCredit(1, 16, 2, 100)
	if got != 16 {
		t.Fatalf("refilled past capacity to %d credits, want 16", got)
	}
}
