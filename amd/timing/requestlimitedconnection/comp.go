// Package requestlimitedconnection provides a zero-latency connection with a
// shared memory-request issue limit. Responses are never throttled.
package requestlimitedconnection

import (
	"fmt"

	"github.com/sarchlab/akita/v5/mem/memprotocol"
	"github.com/sarchlab/akita/v5/messaging"
	"github.com/sarchlab/akita/v5/modeling"
	"github.com/sarchlab/akita/v5/timing"
)

// Spec configures a request-limited connection.
type Spec struct {
	Freq                   timing.Freq   `json:"freq"`
	RequestRateNumerator   int           `json:"request_rate_numerator"`
	RequestRateDenominator int           `json:"request_rate_denominator"`
	BurstRequests          int           `json:"burst_requests"`
	RequestFilter          RequestFilter `json:"request_filter"`
}

// RequestFilter selects which memory request class consumes token credits.
// The zero value preserves the original behavior and limits every AccessReq.
type RequestFilter string

const (
	AllRequests   RequestFilter = ""
	ReadRequests  RequestFilter = "read"
	WriteRequests RequestFilter = "write"
)

// State stores arbitration and token-bucket state.
type State struct {
	NextPortID      int    `json:"next_port_id"`
	RequestCredit   int    `json:"request_credit"`
	LastCreditCycle uint64 `json:"last_credit_cycle"`
}

type ports struct {
	ports   []messaging.Port
	portMap map[messaging.RemotePort]int
}

func (p *ports) addPort(port messaging.Port) {
	p.ports = append(p.ports, port)
	p.portMap[port.AsRemote()] = len(p.ports) - 1
}

func (p *ports) getPortIndex(index int) messaging.Port {
	return p.ports[index]
}

func (p *ports) getPortByName(name messaging.RemotePort) messaging.Port {
	portIndex, found := p.portMap[name]
	if !found {
		panic(fmt.Sprintf("port %s not found", name))
	}

	return p.ports[portIndex]
}

func (p *ports) list() []messaging.Port {
	return p.ports
}

func (p *ports) len() int {
	return len(p.ports)
}

// Comp connects ports while sharing one request budget across all sources and
// destinations.
type Comp struct {
	*modeling.Component[Spec, State, modeling.None]
}

func (c *Comp) mw() *middleware {
	return c.Middlewares()[0].(*middleware)
}

// PlugIn connects a port.
func (c *Comp) PlugIn(port messaging.Port) {
	c.Lock()
	defer c.Unlock()

	c.mw().ports.addPort(port)
	port.SetConnection(c)
}

// Unplug is not supported.
func (c *Comp) Unplug(_ messaging.Port) {
	panic("not implemented")
}

// NotifyAvailable wakes senders after a destination gains buffer space.
func (c *Comp) NotifyAvailable(available messaging.Port) {
	for _, port := range c.mw().ports.list() {
		if port == available {
			continue
		}
		port.NotifyAvailable()
	}
	c.TickNow()
}

// NotifySend starts connection ticking after a port queues a message.
func (c *Comp) NotifySend() {
	c.TickNow()
}

type middleware struct {
	comp  *modeling.Component[Spec, State, modeling.None]
	ports ports
}

func (m *middleware) Tick() bool {
	numPorts := m.ports.len()
	if numPorts == 0 {
		return false
	}

	state := m.comp.State
	spec := m.comp.Spec()
	unlimited := spec.RequestRateNumerator <= 0
	creditDenominator := spec.RequestRateDenominator
	if creditDenominator <= 0 {
		creditDenominator = 1
	}
	requestCredit := m.comp.State.RequestCredit
	if !unlimited {
		capacity := spec.BurstRequests * creditDenominator
		minCapacity := spec.RequestRateNumerator + creditDenominator - 1
		if capacity < minCapacity {
			capacity = minCapacity
		}
		if capacity < creditDenominator {
			capacity = creditDenominator
		}
		currentCycle := spec.Freq.Cycle(m.comp.CurrentTime())
		elapsedCycles := currentCycle - state.LastCreditCycle
		if elapsedCycles == 0 {
			elapsedCycles = 1
		}
		requestCredit = refillRequestCredit(
			requestCredit, capacity, spec.RequestRateNumerator, elapsedCycles,
		)
		(&m.comp.State).LastCreditCycle = currentCycle
	}
	madeProgress := false
	waitingForCredit := false

	for i := range numPorts {
		portID := (i + state.NextPortID) % numPorts
		port := m.ports.getPortIndex(portID)
		madeProgress = m.forwardMany(
			port, &requestCredit, creditDenominator, unlimited,
			&waitingForCredit,
		) || madeProgress
	}

	(&m.comp.State).NextPortID = (state.NextPortID + 1) % numPorts
	(&m.comp.State).RequestCredit = requestCredit

	return madeProgress || waitingForCredit
}

func refillRequestCredit(
	current, capacity, rateNumerator int,
	elapsedCycles uint64,
) int {
	if current >= capacity {
		return capacity
	}

	creditsNeeded := capacity - current
	cyclesToFull := uint64(
		(creditsNeeded + rateNumerator - 1) / rateNumerator,
	)
	if elapsedCycles >= cyclesToFull {
		return capacity
	}

	return current + int(elapsedCycles)*rateNumerator
}

func (m *middleware) forwardMany(
	port messaging.Port,
	requestCredit *int,
	creditDenominator int,
	unlimited bool,
	waitingForCredit *bool,
) bool {
	madeProgress := false

	for {
		head := port.PeekOutgoing()
		if head == nil {
			break
		}

		isRequest := requestConsumesCredit(head, m.comp.Spec().RequestFilter)
		if isRequest && !unlimited && *requestCredit < creditDenominator {
			*waitingForCredit = true
			break
		}

		dstPort := m.ports.getPortByName(head.Meta().Dst)
		if !dstPort.CanDeliver() {
			break
		}

		dstPort.Deliver(head)
		port.RetrieveOutgoing()
		madeProgress = true

		if isRequest && !unlimited {
			*requestCredit -= creditDenominator
		}
	}

	return madeProgress
}

func requestConsumesCredit(msg messaging.Msg, filter RequestFilter) bool {
	if _, ok := msg.(memprotocol.AccessReq); !ok {
		return false
	}

	switch filter {
	case ReadRequests:
		_, ok := msg.(memprotocol.ReadReq)
		return ok
	case WriteRequests:
		_, ok := msg.(memprotocol.WriteReq)
		return ok
	default:
		return true
	}
}
